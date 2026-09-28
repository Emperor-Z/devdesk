# Deploying DevDesk to Cloud Run — free tier, least privilege

## Cost: $0, with a card on file

GCP has no card-free option: even always-free usage needs a billing
account. What you get once one exists:

| Service | Always-free allowance (per month) | DevDesk's use |
|---|---|---|
| Cloud Run | 2M requests, 180k vCPU-s, 360k GB-s | ~25 vCPU-s + ~12 GB-s per question (512Mi) → thousands of questions |
| Artifact Registry | 0.5 GB storage | ~200 MB per image; cleanup policy keeps the latest 2 |
| Secret Manager | 6 active secret versions, 10k accesses | 1 secret, read once per instance start |
| Cloud Build | — | not used: `deploy.sh` builds locally and pushes |

New accounts also get a $300 / 90-day trial credit. When the trial ends,
nothing is charged unless you manually upgrade to a paid account.

The service scales to zero (`minScale: 0`) and is capped at one instance
(`maxScale: 1`), so an idle month costs nothing and a busy one can't fan
out. `deploy.sh setup` can add a **$1 budget alert** (`BILLING_ACCOUNT=...`).
It's an alert, not a spending cap, but at this size it only fires if
something is badly wrong.

### ⚠️ Use a separate project from your Gemini API key

Gemini API usage tiers go by the key's project. **Linking a billing
account to the project that owns your AI Studio key moves the key from
the free tier to paid Tier 1.** So create a new GCP project for Cloud
Run, and leave the key's project without billing. The key itself works
from any project: it's just a string stored in Secret Manager.

## Identity model

| Principal | Gets | Why |
|---|---|---|
| `devdesk-runtime` SA (the running container) | `roles/secretmanager.secretAccessor` on **one secret** only | reads the Gemini key; no project-level roles at all |
| You (the caller) | `roles/run.invoker` on **this service** only | the only way through Cloud Run's IAM check; there is no `allUsers` binding |
| You (the deployer) | project owner/editor on your own project | one-off setup, not a runtime identity |

The app has no auth code: Cloud Run IAM is the boundary, and an
unauthenticated request gets a `403` before it reaches the container. The
key never sits in the image, the repo, or a plain env var in the service
spec. It's injected from Secret Manager at instance start.

## Steps

1. **Account + project** (browser): sign in at console.cloud.google.com,
   start the free trial (this creates the billing account), then create a
   **new** project, e.g. `devdesk-run`. Note its project ID and the billing
   account ID (Billing → Account management, format `XXXXXX-XXXXXX-XXXXXX`).

2. **gcloud CLI** (once):
   ```bash
   # Arch: yay -S google-cloud-cli    — or the tarball from cloud.google.com/sdk
   gcloud auth login
   ```

3. **Setup** (once): enables APIs, creates the runtime SA, stores the key
   from `.env` as a secret, creates the registry and its cleanup policy,
   and adds the budget alert.
   ```bash
   export PROJECT_ID=devdesk-run REGION=europe-west1
   BILLING_ACCOUNT=XXXXXX-XXXXXX-XXXXXX deploy/deploy.sh setup
   ```

4. **Deploy** (each release): gates on the retrieval eval (the index is
   baked into the image), builds, pushes an image tagged with the commit,
   rolls out, and grants *you* invoker.
   ```bash
   deploy/deploy.sh deploy
   ```

5. **Use it:**
   ```bash
   deploy/deploy.sh ask "What slash commands does the Ares REPL support?"
   ```

The service spec is `deploy/cloudrun_service.yaml`. `deploy.sh` fills in
`PROJECT_ID`/`REGION`/the image tag and applies it with
`gcloud run services replace`.

## Deliberate limits

- **`maxScale: 1`**: the rate limiter is per process and the Gemini
  free-tier quota is per project, so more instances would each pace
  independently and exceed it together.
- **Index is a snapshot.** Tenant source repos aren't in the image, so
  `git_status_lookup` reports "no local repo configured" in the cloud,
  since it's a local-only tool by nature. Re-run ingest and redeploy to
  refresh.
- **AI Studio key, not Vertex AI.** Vertex would drop the secret entirely
  (the runtime SA would get `roles/aiplatform.user` instead), but it has
  no free tier. The switch is `GOOGLE_GENAI_USE_VERTEXAI=TRUE` plus
  passing project/location instead of `api_key` in `rag/embeddings.py`.
