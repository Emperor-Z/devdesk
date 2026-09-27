# Deploying DevDesk to Cloud Run (least-privilege)

Everything here stays within Cloud Run / Artifact Registry / Secret
Manager free-tier limits for personal use. It does require a GCP project
with billing enabled (Cloud Run won't deploy without it, even at $0 usage).

## Identity model

| Principal | Gets | Why |
|---|---|---|
| `devdesk-runtime` SA (the running container) | `roles/secretmanager.secretAccessor` on **one secret** only | reads the Gemini key; no project-level roles at all |
| You (the caller) | `roles/run.invoker` on **this service** only | the only way through Cloud Run's IAM check — there is no `allUsers` binding |
| Deployer (you, from a workstation) | normal project editor-level access | one-off setup, not a runtime identity |

The app itself has no auth code: Cloud Run IAM is the boundary. The Gemini
key never sits in the image, the repo, or a plain env var in the service
spec — it's injected from Secret Manager at instance start.

## One-time setup

```bash
PROJECT_ID=your-project
REGION=europe-west1   # Tier-1 pricing: cheapest if usage ever exceeds the free tier
gcloud config set project "$PROJECT_ID"

gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
    secretmanager.googleapis.com cloudbuild.googleapis.com

# Runtime identity — created with no roles.
gcloud iam service-accounts create devdesk-runtime \
    --display-name="DevDesk Cloud Run runtime"

# Gemini API key as a secret, readable by that SA only.
printf '%s' "$GOOGLE_API_KEY" | gcloud secrets create devdesk-gemini-api-key \
    --replication-policy=automatic --data-file=-
gcloud secrets add-iam-policy-binding devdesk-gemini-api-key \
    --member="serviceAccount:devdesk-runtime@$PROJECT_ID.iam.gserviceaccount.com" \
    --role=roles/secretmanager.secretAccessor

gcloud artifacts repositories create devdesk \
    --repository-format=docker --location="$REGION"
```

## Build and deploy

```bash
python -m devdesk.rag.ingest            # refresh data/.chroma — it's baked into the image
python eval/run_eval.py --mode retrieval --fail-under 1.0   # don't ship a regressed index

IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/devdesk/devdesk:latest"
gcloud builds submit --tag "$IMAGE"

sed -e "s/PROJECT_ID/$PROJECT_ID/g" -e "s/REGION/$REGION/g" \
    deploy/cloudrun_service.yaml > /tmp/devdesk-service.yaml
gcloud run services replace /tmp/devdesk-service.yaml --region "$REGION"

# Let yourself (and only yourself) call it.
gcloud run services add-iam-policy-binding devdesk --region "$REGION" \
    --member="user:$(gcloud config get-value account)" --role=roles/run.invoker
```

## Calling it

```bash
URL=$(gcloud run services describe devdesk --region "$REGION" --format='value(status.url)')
curl -s "$URL/ask" \
    -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
    -H 'content-type: application/json' \
    -d '{"question": "What slash commands does the Ares REPL support?"}'
```

An unauthenticated request gets a `403` from Cloud Run before it ever
reaches the container.

## Deliberate limits

- **`maxScale: 1`** — the rate limiter is per process and the Gemini
  free-tier quota is per project; more instances would each pace
  independently and exceed it together.
- **Index is a snapshot.** Tenant source repos aren't in the image, so
  `git_status_lookup` reports "no local repo configured" in the cloud —
  it's a local-only tool by nature.
- **AI Studio key, not Vertex AI.** Vertex would drop the secret entirely
  (the runtime SA would get `roles/aiplatform.user` instead), but it isn't
  free-tier. The switch is `GOOGLE_GENAI_USE_VERTEXAI=TRUE` plus passing
  project/location instead of `api_key` in `rag/embeddings.py`.
