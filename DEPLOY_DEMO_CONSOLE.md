# Demo console — command sheet

Copy-paste commands for the Fly.io deployment of the demo console
(`Dockerfile` + `app/` + `deploy_bundle/`). The full pipeline is a different
deployment — see `DEPLOY.md`.

App: **real-time-fraud-detect-console** · Region: **sin** · Config: `fly.toml`
Live: https://real-time-fraud-detect-console.fly.dev

> `--ha=false` on every deploy. Without it Fly creates a second machine and the
> bill doubles for redundancy a demo does not need.

---

## One-time setup

```bash
brew install flyctl
fly auth login
fly auth whoami
fly apps create real-time-fraud-detect-console
```

---

## Deploy / update

```bash
fly deploy --ha=false
```

Covers any change to `app/`, `features.py`, `web/src/`, `deploy_bundle/`,
`demo_data/` or `Dockerfile`. The frontend is built inside the image — no local
`npm run build` needed to deploy.

---

## Ship a new model

The model is frozen into the image, so re-freeze it before deploying.

```bash
docker compose up -d mlflow
export MLFLOW_TRACKING_URI=http://localhost:5001

python train_fraud_model.py

python scripts/promote_model.py --list
python scripts/promote_model.py --version 7 --alias champion --min-auc 0.85

python scripts/export_for_deploy.py          # registry -> deploy_bundle/
git diff --stat deploy_bundle/               # confirm it actually changed

fly deploy --ha=false
```

Skipping `export_for_deploy.py` redeploys the old model with new code.

---

## Ship different demo transactions

```bash
python scripts/build_demo_data.py --rows 15000
cat demo_data/demo_metadata.json
fly deploy --ha=false
```

---

## Test before deploying

```bash
# Native — needs the frontend built, since app/main.py serves web/dist
cd web && npm run build && cd ..
python3 -m uvicorn app.main:app --port 8000

# Or exactly what Fly will run
docker build -t fraud-console:test .
docker run --rm -d --name console-test -p 8899:8080 fraud-console:test
curl -s localhost:8899/api/health
curl -sN --max-time 2 "localhost:8899/api/stream?rate=250&loop=true&shuffle=true" | head -5
docker rm -f console-test
```

Expected on startup:

```
[APP] Scoring engine ready | model v6 | 439 features | threshold 0.7933
[APP] Loaded 12000 demo transactions
[APP] Precomputed 12000 scores | fraud=393
```

---

## Check / debug

```bash
fly status                  # machine count, state, health checks
fly logs                    # live tail
fly logs --no-tail | tail -50
fly open                    # open the site
fly dashboard               # metrics and billing in the browser
fly ssh console             # shell inside the running machine

curl -s https://real-time-fraud-detect-console.fly.dev/api/health
curl -s https://real-time-fraud-detect-console.fly.dev/api/info | python3 -m json.tool | head -20
```

---

## Roll back

```bash
fly releases --image       # lists each release with its image reference
fly deploy --ha=false --image registry.fly.io/real-time-fraud-detect-console:deployment-<older-id>
```

Skips the build, so recovery is under a minute.

---

## Change the machine

```bash
fly scale show
fly scale count 1           # undo an accidental second machine
fly scale memory 512        # ~$4.04/mo instead of ~$6.50
fly scale vm shared-cpu-1x  # halves usable concurrent viewers — measure first
```

`fly scale` changes the running machine now; edit `fly.toml` too, or the next
`fly deploy` puts it back.

Region (edit `fly.toml`, then deploy):

```toml
primary_region = "sin"      # Singapore. iad=Virginia, fra=Frankfurt. bom is deprecated.
```

---

## Cost control

Always-on is the whole cost. To sleep when idle — ~$0.20/mo, 20-30s wake for the
first visitor — edit `fly.toml`:

```toml
auto_stop_machines = true
min_machines_running = 0
```

then `fly deploy --ha=false`.

Stop paying entirely:

```bash
fly apps destroy real-time-fraud-detect-console
```

Billing stops immediately. Redeploy later with the One-time setup block.

---

## Custom domain

```bash
fly certs add fraud.example.com      # prints the DNS records to add
fly certs check fraud.example.com    # certificate + DNS status
fly certs list
```

---

## Before every deploy

```bash
git status                  # fly deploy ships what is on disk, not what is committed
git push                    # let CI run the parity tests first
fly deploy --ha=false
```

CI catches the train/serve contract breaking, which produces no error at
runtime — just silently wrong scores.
