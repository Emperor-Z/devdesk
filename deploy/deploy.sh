#!/usr/bin/env bash
# DevDesk -> Cloud Run, sized to stay inside GCP's always-free tier.
#
#   deploy/deploy.sh setup           # once: APIs, runtime SA, secret, registry, budget
#   deploy/deploy.sh deploy          # build locally, push, roll out, grant yourself invoker
#   deploy/deploy.sh ask "question"  # authenticated call to the live service
#   deploy/deploy.sh local           # run the image as Cloud Run would, smoke-test it
#
# Env:
#   PROJECT_ID       required. The Cloud Run project — NOT the project that owns
#                    your Gemini API key (billing there moves the key off the
#                    free tier; see deploy/iam_setup.md).
#   REGION           default europe-west1
#   BILLING_ACCOUNT  optional; setup creates a $1 budget alert on it
#   GOOGLE_API_KEY   optional; setup reads it from .env otherwise
#   SKIP_EVAL=1      skip the retrieval-eval gate before deploy
#   DRY_RUN=1        print every gcloud/push command instead of running it;
#                    the build, eval gate and manifest render still run for
#                    real. PROJECT_ID defaults to a stand-in, no gcloud needed.
set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN="${DRY_RUN:-}"
if [[ "$DRY_RUN" == 1 ]]; then
  PROJECT_ID="${PROJECT_ID:-devdesk-standin}"
elif [[ "${1:-}" != local ]]; then
  : "${PROJECT_ID:?set PROJECT_ID to the Cloud Run project (not the project that owns the Gemini key)}"
fi
PROJECT_ID="${PROJECT_ID:-devdesk-standin}"
REGION="${REGION:-europe-west1}"
PYTHON="${PYTHON:-.venv/bin/python}"
SA="devdesk-runtime@${PROJECT_ID}.iam.gserviceaccount.com"
SECRET=devdesk-gemini-api-key
REPO=devdesk
SERVICE=devdesk
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/devdesk"
G=(gcloud --project="$PROJECT_ID" --quiet)

# Side-effecting commands go through run(); in a dry run they're printed.
run() {
  if [[ "$DRY_RUN" == 1 ]]; then
    { printf '[dry-run]'; printf ' %q' "$@"; echo; } >&2
  else
    "$@"
  fi
}

# Existence checks: a dry run assumes nothing exists yet, so it shows the
# full first-time setup.
exists() {
  [[ "$DRY_RUN" == 1 ]] && return 1
  "$@" >/dev/null 2>&1
}

api_key() {
  if [[ -n "${GOOGLE_API_KEY:-}" ]]; then
    printf '%s' "$GOOGLE_API_KEY"
  else
    grep -E '^GOOGLE_API_KEY=' .env 2>/dev/null | head -1 | cut -d= -f2- | sed -E "s/^[\"']|[\"']\$//g"
  fi
}

budget() {
  local name="devdesk-1usd"
  if [[ "$DRY_RUN" != 1 ]] && gcloud billing budgets list --billing-account="$BILLING_ACCOUNT" \
      --format='value(displayName)' 2>/dev/null | grep -qx "$name"; then
    echo "Budget alert '$name' already exists"
    return
  fi
  # An alert, not a cap: GCP budgets notify, they don't stop spending.
  run gcloud billing budgets create --billing-account="$BILLING_ACCOUNT" \
    --display-name="$name" --budget-amount=1USD \
    --threshold-rule=percent=0.5 --threshold-rule=percent=1.0 \
    --filter-projects="projects/${PROJECT_ID}"
}

setup() {
  run "${G[@]}" services enable run.googleapis.com artifactregistry.googleapis.com \
    secretmanager.googleapis.com billingbudgets.googleapis.com

  # Runtime identity, created with no roles at all.
  exists "${G[@]}" iam service-accounts describe "$SA" ||
    run "${G[@]}" iam service-accounts create devdesk-runtime \
      --display-name="DevDesk Cloud Run runtime"

  if ! exists "${G[@]}" secrets describe "$SECRET"; then
    local key
    key="$(api_key)"
    [[ -n "$key" ]] || { echo "No GOOGLE_API_KEY in env or .env" >&2; exit 1; }
    printf '%s' "$key" | run "${G[@]}" secrets create "$SECRET" \
      --replication-policy=automatic --data-file=-
  fi
  # Its only permission: read this one secret.
  run "${G[@]}" secrets add-iam-policy-binding "$SECRET" \
    --member="serviceAccount:${SA}" --role=roles/secretmanager.secretAccessor >/dev/null

  exists "${G[@]}" artifacts repositories describe "$REPO" --location="$REGION" ||
    run "${G[@]}" artifacts repositories create "$REPO" \
      --repository-format=docker --location="$REGION"
  run "${G[@]}" artifacts repositories set-cleanup-policies "$REPO" \
    --location="$REGION" --policy=deploy/ar_cleanup_policy.json

  if [[ -n "${BILLING_ACCOUNT:-}" ]]; then
    budget
  else
    echo "Skipping budget alert (set BILLING_ACCOUNT to create one)"
  fi
  echo "Setup done. Next: PROJECT_ID=$PROJECT_ID deploy/deploy.sh deploy"
}

deploy() {
  [[ -d data/.chroma ]] || { echo "No index: run python -m devdesk.rag.ingest first" >&2; exit 1; }
  if [[ "${SKIP_EVAL:-}" != 1 ]]; then
    # The index is baked into the image — don't ship a regressed one.
    "$PYTHON" eval/run_eval.py --mode retrieval --fail-under 1.0 --no-save
  fi

  local tag tmp url account
  tag="$(image_tag)"
  run gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
  # Built locally and pushed: no Cloud Build API, bucket, or build SA needed.
  docker build -t "${IMAGE}:${tag}" .
  run docker push "${IMAGE}:${tag}"

  tmp="$(mktemp)"
  trap 'rm -f "$tmp"' RETURN
  render_manifest "$tag" >"$tmp"
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "[dry-run] rendered manifest:"; sed 's/^/    /' "$tmp"
  fi
  run "${G[@]}" run services replace "$tmp" --region="$REGION"

  # Only you can invoke it: no allUsers binding, ever.
  if [[ "$DRY_RUN" == 1 ]]; then
    account="you@example.com"
  else
    account="$(gcloud config get-value account 2>/dev/null)"
  fi
  run "${G[@]}" run services add-iam-policy-binding "$SERVICE" --region="$REGION" \
    --member="user:${account}" --role=roles/run.invoker >/dev/null

  if [[ "$DRY_RUN" == 1 ]]; then
    echo "Dry run done: image ${IMAGE}:${tag} built, nothing sent to GCP."
    return
  fi
  url="$("${G[@]}" run services describe "$SERVICE" --region="$REGION" --format='value(status.url)')"
  echo "Deployed ${tag} -> ${url}"
}

image_tag() {
  printf '%s%s' "$(git rev-parse --short HEAD)" "$(git diff --quiet HEAD || echo -dirty)"
}

render_manifest() {
  local out
  out="$(sed -e "s|PROJECT_ID|${PROJECT_ID}|g" -e "s|REGION|${REGION}|g" \
      -e "s|devdesk/devdesk:latest|devdesk/devdesk:${1}|" \
      deploy/cloudrun_service.yaml)"
  if grep -v '^#' <<<"$out" | grep -qE 'PROJECT_ID|REGION'; then
    echo "Unfilled placeholder in rendered manifest" >&2
    return 1
  fi
  printf '%s\n' "$out"
}

# The container as Cloud Run runs it: same image, port, memory/CPU limits and
# plain env from the manifest. The key comes from .env in place of Secret
# Manager. Smoke-tests /healthz and one /ask, then stops the container.
local_run() {
  [[ -d data/.chroma ]] || { echo "No index: run python -m devdesk.rag.ingest first" >&2; exit 1; }
  local key tag name=devdesk-local port="${LOCAL_PORT:-8080}" envs=() i
  key="$(api_key)"
  [[ -n "$key" ]] || { echo "No GOOGLE_API_KEY in env or .env" >&2; exit 1; }
  tag="$(image_tag)"
  docker build -t "devdesk-local:${tag}" .
  mapfile -t envs < <("$PYTHON" - <<'PY'
import yaml
spec = yaml.safe_load(open("deploy/cloudrun_service.yaml"))
container = spec["spec"]["template"]["spec"]["containers"][0]
for env in container.get("env", []):
    if "value" in env:
        print(f"{env['name']}={env['value']}")
PY
)
  local args=(--rm -d --name "$name" -p "127.0.0.1:${port}:8080" --memory 512m --cpus 1
    -e GOOGLE_API_KEY)
  for i in "${envs[@]}"; do args+=(-e "$i"); done
  docker rm -f "$name" >/dev/null 2>&1 || true
  GOOGLE_API_KEY="$key" docker run "${args[@]}" "devdesk-local:${tag}" >/dev/null
  trap 'docker rm -f devdesk-local >/dev/null 2>&1 || true' EXIT

  for i in $(seq 1 60); do
    curl -fsS "http://127.0.0.1:${port}/healthz" >/dev/null 2>&1 && break
    sleep 1
  done
  echo "healthz: $(curl -fsS "http://127.0.0.1:${port}/healthz")"
  echo "ask:"
  curl -fsS -m 300 "http://127.0.0.1:${port}/ask" -H 'content-type: application/json' \
    -d '{"question": "What slash commands does the Ares REPL support?"}'
  echo
  echo "memory after ask: $(docker stats --no-stream --format '{{.MemUsage}}' "$name")"
}

ask() {
  local url body
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "Nothing to ask in a dry run; use deploy/deploy.sh local" >&2
    exit 1
  fi
  url="$("${G[@]}" run services describe "$SERVICE" --region="$REGION" --format='value(status.url)')"
  body="$("$PYTHON" -c 'import json,sys; print(json.dumps({"question": sys.argv[1]}))' "$1")"
  curl -sS -m 300 "${url}/ask" \
    -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
    -H 'content-type: application/json' -d "$body"
  echo
}

case "${1:-}" in
  setup) setup ;;
  deploy) deploy ;;
  ask) shift; ask "${1:?usage: deploy/deploy.sh ask \"question\"}" ;;
  local) local_run ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
