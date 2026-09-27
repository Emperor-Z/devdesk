#!/usr/bin/env bash
# DevDesk -> Cloud Run, sized to stay inside GCP's always-free tier.
#
#   deploy/deploy.sh setup           # once: APIs, runtime SA, secret, registry, budget
#   deploy/deploy.sh deploy          # build locally, push, roll out, grant yourself invoker
#   deploy/deploy.sh ask "question"  # authenticated call to the live service
#
# Env:
#   PROJECT_ID       required. The Cloud Run project — NOT the project that owns
#                    your Gemini API key (billing there moves the key off the
#                    free tier; see deploy/iam_setup.md).
#   REGION           default europe-west1
#   BILLING_ACCOUNT  optional; setup creates a $1 budget alert on it
#   GOOGLE_API_KEY   optional; setup reads it from .env otherwise
#   SKIP_EVAL=1      skip the retrieval-eval gate before deploy
set -euo pipefail
cd "$(dirname "$0")/.."

: "${PROJECT_ID:?set PROJECT_ID to the Cloud Run project (not the project that owns the Gemini key)}"
REGION="${REGION:-europe-west1}"
PYTHON="${PYTHON:-.venv/bin/python}"
SA="devdesk-runtime@${PROJECT_ID}.iam.gserviceaccount.com"
SECRET=devdesk-gemini-api-key
REPO=devdesk
SERVICE=devdesk
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/devdesk"
G=(gcloud --project="$PROJECT_ID" --quiet)

api_key() {
  if [[ -n "${GOOGLE_API_KEY:-}" ]]; then
    printf '%s' "$GOOGLE_API_KEY"
  else
    grep -E '^GOOGLE_API_KEY=' .env 2>/dev/null | head -1 | cut -d= -f2- | sed -E "s/^[\"']|[\"']\$//g"
  fi
}

budget() {
  local name="devdesk-1usd"
  if gcloud billing budgets list --billing-account="$BILLING_ACCOUNT" \
      --format='value(displayName)' 2>/dev/null | grep -qx "$name"; then
    echo "Budget alert '$name' already exists"
    return
  fi
  # An alert, not a cap: GCP budgets notify, they don't stop spending.
  gcloud billing budgets create --billing-account="$BILLING_ACCOUNT" \
    --display-name="$name" --budget-amount=1USD \
    --threshold-rule=percent=0.5 --threshold-rule=percent=1.0 \
    --filter-projects="projects/${PROJECT_ID}"
}

setup() {
  "${G[@]}" services enable run.googleapis.com artifactregistry.googleapis.com \
    secretmanager.googleapis.com billingbudgets.googleapis.com

  # Runtime identity, created with no roles at all.
  "${G[@]}" iam service-accounts describe "$SA" >/dev/null 2>&1 ||
    "${G[@]}" iam service-accounts create devdesk-runtime \
      --display-name="DevDesk Cloud Run runtime"

  if ! "${G[@]}" secrets describe "$SECRET" >/dev/null 2>&1; then
    local key
    key="$(api_key)"
    [[ -n "$key" ]] || { echo "No GOOGLE_API_KEY in env or .env" >&2; exit 1; }
    printf '%s' "$key" | "${G[@]}" secrets create "$SECRET" \
      --replication-policy=automatic --data-file=-
  fi
  # Its only permission: read this one secret.
  "${G[@]}" secrets add-iam-policy-binding "$SECRET" \
    --member="serviceAccount:${SA}" --role=roles/secretmanager.secretAccessor >/dev/null

  "${G[@]}" artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1 ||
    "${G[@]}" artifacts repositories create "$REPO" \
      --repository-format=docker --location="$REGION"
  "${G[@]}" artifacts repositories set-cleanup-policies "$REPO" \
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

  local tag tmp url
  tag="$(git rev-parse --short HEAD)$(git diff --quiet HEAD || echo -dirty)"
  gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
  # Built locally and pushed: no Cloud Build API, bucket, or build SA needed.
  docker build -t "${IMAGE}:${tag}" .
  docker push "${IMAGE}:${tag}"

  tmp="$(mktemp)"
  trap 'rm -f "$tmp"' RETURN
  sed -e "s|PROJECT_ID|${PROJECT_ID}|g" -e "s|REGION|${REGION}|g" \
      -e "s|devdesk/devdesk:latest|devdesk/devdesk:${tag}|" \
      deploy/cloudrun_service.yaml >"$tmp"
  "${G[@]}" run services replace "$tmp" --region="$REGION"

  # Only you can invoke it: no allUsers binding, ever.
  "${G[@]}" run services add-iam-policy-binding "$SERVICE" --region="$REGION" \
    --member="user:$(gcloud config get-value account 2>/dev/null)" \
    --role=roles/run.invoker >/dev/null

  url="$("${G[@]}" run services describe "$SERVICE" --region="$REGION" --format='value(status.url)')"
  echo "Deployed ${tag} -> ${url}"
}

ask() {
  local url body
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
  *) sed -n '2,16p' "$0"; exit 1 ;;
esac
