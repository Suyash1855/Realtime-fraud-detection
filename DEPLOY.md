# Deploy — full pipeline on a Hetzner Cloud VM

The whole stack on one server: Kafka, Redis, MLflow (Postgres + MinIO),
Prometheus, Grafana, the scorer and the read API, with Caddy terminating TLS in
front. `RUNBOOK.md` is the local equivalent; this is the same compose file plus
`docker-compose.prod.yml`.

The demo console is **not** part of this. It stays on Fly.io (`fly.toml`,
root `Dockerfile`, commands in `DEPLOY_DEMO_CONSOLE.md`) — it is self-contained and
needs none of the infrastructure below.

**What ends up public:** `https://api.<domain>` (read + ingest API, API-key
protected) and `https://grafana.<domain>` (Grafana's own login). Everything else
binds to `127.0.0.1` on the server and is reached over an SSH tunnel.

Prices below are approximate — check current Hetzner rates before committing.

---

## Step 0 — Before you touch Hetzner

**Push the deployment files.** Four of them are still untracked, and a `git
clone` on the server would come up without them:

```bash
git add Dockerfile.pipeline Dockerfile.mlflow docker-compose.prod.yml Caddyfile \
        .env.example DEPLOY.md scripts/migrate_mlflow_store.py \
        docker-compose.yml consumer.py API/main.py
git commit -m "Add production compose overlay, Caddy proxy and deploy guide"
git push
```

**Pick a domain.** You need one you control DNS for. Two subdomains:
`api.<domain>` and `grafana.<domain>`.

---

## Step 1 — Create the server

In the [Hetzner Cloud console](https://console.hetzner.cloud): **New project** →
**Add server**.

| Setting | Value | Why |
|---|---|---|
| Location | Falkenstein / Nuremberg (EU) or Ashburn (US) | Nearest to whoever opens the link |
| Image | Ubuntu 24.04 | |
| Type | **CX42** — 8 vCPU, 16 GB, 160 GB (~€17/mo) | The `mem_limit`s in compose sum to ~8.6 GB |
| SSH key | Add your public key | Password login gets disabled in step 4 |
| Firewall | Create one, see step 3 | |

**CX32** (4 vCPU / 8 GB, ~€8/mo) is the floor — it runs, with no headroom for
the producer replaying the dataset at full rate. **Do not pick a CAX (ARM)
type** unless you first confirm `confluentinc/cp-kafka:7.5.0`,
`redis/redis-stack` and `provectuslabs/kafka-ui` all publish arm64 tags; CX and
CPX are x86 and avoid the question.

---

## Step 2 — DNS

Point both subdomains at the server's IPv4 address:

```
api.<domain>      A    <server-ip>
grafana.<domain>  A    <server-ip>
```

Do this **before** the first start. Caddy orders certificates over HTTP-01 on
first boot, and a record that does not resolve yet fails the order and enters a
retry backoff.

Check propagation: `dig +short api.<domain>`

---

## Step 3 — Firewall

A Hetzner Cloud Firewall, attached to the server. Inbound rules, everything else
denied:

| Port | Protocol | Source |
|---|---|---|
| 22 | TCP | your IP, or `0.0.0.0/0` if your address moves |
| 80 | TCP | `0.0.0.0/0` (ACME challenge + HTTP→HTTPS redirect) |
| 443 | TCP | `0.0.0.0/0` |

Use the cloud firewall rather than `ufw`. Docker writes its own iptables rules
for published ports and those bypass ufw entirely — a ufw rule would have given
you the *appearance* of a closed Kafka port. The real protection here is that
every service except Caddy binds to `127.0.0.1` via `BIND_ADDRESS`.

---

## Step 4 — First login and hardening

```bash
ssh root@<server-ip>

# A non-root user to run everything as
adduser --disabled-password --gecos "" deploy
mkdir -p /home/deploy/.ssh
cp /root/.ssh/authorized_keys /home/deploy/.ssh/
chown -R deploy:deploy /home/deploy/.ssh
chmod 700 /home/deploy/.ssh && chmod 600 /home/deploy/.ssh/authorized_keys
usermod -aG sudo deploy
# Used only by sudo -- SSH stays key-only below. Without it sudo has nothing to check.
passwd deploy

# Swap — 16 GB is enough for steady state, not for a JVM heap spike during a
# full-rate replay while the consumer is also loading the model.
fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

# Security updates without a prompt
apt update && apt install -y unattended-upgrades
printf 'APT::Periodic::Update-Package-Lists "1";\nAPT::Periodic::Unattended-Upgrade "1";\n' \
  > /etc/apt/apt.conf.d/20auto-upgrades

# Keys only. A drop-in rather than editing sshd_config: sshd takes the first value
# it reads, and cloud-init's 50-cloud-init.conf would otherwise win.
printf 'PasswordAuthentication no\nPermitRootLogin prohibit-password\n' \
  > /etc/ssh/sshd_config.d/00-hardening.conf
sshd -t && systemctl restart ssh
sshd -T | grep -E '^(passwordauthentication|permitrootlogin) '
```

Open a **second terminal** and confirm `ssh deploy@<server-ip>` works before
closing the root session.

---

## Step 5 — Install Docker

As `deploy`:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker deploy
exit    # group membership applies on next login
```

Log back in and check: `docker compose version`

---

## Step 6 — Get the code

If the repo is private, create a read-only deploy key:

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
# → GitHub → repo → Settings → Deploy keys → Add, read-only
git clone git@github.com:Suyash1855/Realtime-fraud-detection.git fraud-detect-system
```

Public repo: `git clone https://github.com/Suyash1855/Realtime-fraud-detection.git fraud-detect-system`

Clone into `fraud-detect-system`, not the repo's own name. Compose derives the
project name from the directory and the volumes follow from it
(`fraud-detect-system_mlflow-db`, …), so keeping it identical to the local
directory means every command in `RUNBOOK.md` works unchanged on the server.
Renaming it later orphans every volume.

```bash
cd ~/fraud-detect-system
```

---

## Step 7 — Write `.env`

`.env` is gitignored, so it does not exist on the server yet. Generate it with
real secrets — hex only, because `REDIS_PASSWORD` is interpolated into
`REDIS_ARGS` as a shell word:

```bash
umask 077
cat > .env <<EOF
REDIS_PASSWORD=$(openssl rand -hex 24)
POSTGRES_PASSWORD=$(openssl rand -hex 24)
MINIO_ROOT_USER=fraudminio
MINIO_ROOT_PASSWORD=$(openssl rand -hex 24)
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=$(openssl rand -hex 24)
API_KEY=$(openssl rand -hex 32)

KAFKA_ADVERTISED_HOST=localhost
BIND_ADDRESS=127.0.0.1
API_BIND_ADDRESS=127.0.0.1

DOMAIN=<your-domain>
ACME_EMAIL=<your-email>
EOF
chmod 600 .env
grep -E 'API_KEY|GRAFANA_ADMIN_PASSWORD' .env    # save these two in your password manager
```

Then edit `DOMAIN` and `ACME_EMAIL` to real values — the heredoc wrote the
placeholders literally.

Two of these are worth understanding rather than copying:

- `KAFKA_ADVERTISED_HOST=localhost` is correct **because nothing outside the
  Docker network is a Kafka client here.** Producer, consumer and API all reach
  the broker on the INTERNAL listener (`kafka:29092`). The EXTERNAL listener on
  9092 stays bound to loopback and is only used if you tunnel to it.
- `API_BIND_ADDRESS=127.0.0.1` is what closes the plaintext port 8200. Caddy
  reaches the API over the compose network, not through it.

---

## Step 8 — Build and start

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

First run takes 5–10 minutes: it builds `fraud-pipeline` (pandas, xgboost,
mlflow) and `fraud-mlflow`, and pulls eight images.

Typing both `-f` flags every time gets old. Optional, on the server only:

```bash
echo 'COMPOSE_FILE=docker-compose.yml:docker-compose.prod.yml' >> .env
```

after which plain `docker compose up -d` includes the overlay. Every command
below shows the explicit form.

---

## Step 9 — Verify

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
```

Every long-running service `healthy`; `kafka-init` and `minio-init` `exited (0)`.
`fraud-api` takes ~75s to report healthy on a cold start.

```bash
export API_KEY=$(grep '^API_KEY=' .env | cut -d= -f2)

# TLS — certificate should be issued within a minute of first start
curl -s https://api.<domain>/health

# Authenticated read
curl -s -H "X-API-Key: $API_KEY" https://api.<domain>/stats

# The proxy blocks /metrics on purpose; Prometheus scrapes it in-network
curl -s -o /dev/null -w '%{http_code}\n' https://api.<domain>/metrics    # 404
```

If the certificate has not appeared: `docker compose -f docker-compose.yml -f
docker-compose.prod.yml logs caddy`.

`/stats` returns zeros at this point. Nothing has been scored yet, and the
consumer is still failing to resolve a model — that is step 10.

---

## Step 10 — Seed the model registry

The registry is a fresh Postgres and an empty MinIO bucket, so
`models:/fraud-xgboost@champion` resolves to nothing and `fraud-consumer` will
not score. Two ways to fix it.

### Option A — migrate your local runs (recommended)

`scripts/migrate_mlflow_store.py` copies runs and registered versions from one
tracking server to another, preserving version numbers. Three terminals **on
your laptop**:

```bash
# 1. Serve the old file store read-only
docker run --rm -d --name mlflow-legacy -p 5002:5000 \
  -v "$PWD/mlruns:/mlflow/mlruns:ro" -v "$PWD/mlartifacts:/mlflow/mlartifacts:ro" \
  ghcr.io/mlflow/mlflow:v3.1.4 mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri file:///mlflow/mlruns \
  --artifacts-destination /mlflow/mlartifacts --serve-artifacts

# 2. Tunnel the server's MLflow to local 5555 (5001 may be taken locally)
ssh -N -L 5555:127.0.0.1:5001 deploy@<server-ip>

# 3. Copy
python scripts/migrate_mlflow_store.py --source http://localhost:5002 \
                                       --dest   http://localhost:5555
```

Then promote, through the same tunnel:

```bash
MLFLOW_TRACKING_URI=http://localhost:5555 python scripts/promote_model.py --list
MLFLOW_TRACKING_URI=http://localhost:5555 python scripts/promote_model.py \
        --version 6 --alias champion --min-auc 0.85
```

Run ids change (the destination assigns its own); version numbers do not, which
is why `--version 6` still means the model in `deploy_bundle/manifest.json`.

Restart the scorer so it re-resolves:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml restart fraud-consumer
docker compose -f docker-compose.yml -f docker-compose.prod.yml logs --tail 30 fraud-consumer
```

Look for the model version and threshold in the startup lines.

### Option B — retrain on the server

Heavier: needs `data/` uploaded (step 11) and the training dependencies, which
are deliberately *not* in `Dockerfile.pipeline`.

```bash
sudo apt install -y python3-venv
python3 -m venv ~/venv && ~/venv/bin/pip install -r requirements.txt
MLFLOW_TRACKING_URI=http://localhost:5001 ~/venv/bin/python train_fraud_model.py
MLFLOW_TRACKING_URI=http://localhost:5001 ~/venv/bin/python scripts/promote_model.py --list
```

~10 minutes on 590k rows, and it will use most of the RAM. Stop the producer
first if one is running.

---

## Step 11 — Send traffic

### From the dataset

The producer mounts `data/` read-only; the CSVs are gitignored, so upload them
(1.3 GB, once):

```bash
rsync -avz --progress data/ deploy@<server-ip>:~/fraud-detect-system/data/
```

Then, on the server:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  --profile load run --rm fraud-producer
```

### Or over the API, no upload needed

```bash
curl -X POST https://api.<domain>/transactions \
  -H "X-API-Key: $API_KEY" -H 'Content-Type: application/json' \
  -d '{"TransactionID": 1, "TransactionDT": 86400, "TransactionAmt": 120.5, ...}'
```

`TransactionDT` is required and is the dataset's integer offset, not a
timestamp. Watch the results land:

```bash
curl -s -H "X-API-Key: $API_KEY" https://api.<domain>/stats
curl -s -H "X-API-Key: $API_KEY" https://api.<domain>/frauds
```

---

## Step 12 — Reach the private UIs

Grafana is public at `https://grafana.<domain>` (log in with
`GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD`). Everything else is loopback
only — one tunnel covers them all:

```bash
ssh -N -L 5555:127.0.0.1:5001 \
       -L 9090:127.0.0.1:9090 \
       -L 8080:127.0.0.1:8080 \
       -L 8001:127.0.0.1:8001 \
       -L 9001:127.0.0.1:9001 \
       deploy@<server-ip>
```

| Local URL | Service |
|---|---|
| http://localhost:5555 | MLflow (5001 on the server) |
| http://localhost:9090 | Prometheus |
| http://localhost:8080 | kafka-ui |
| http://localhost:8001 | RedisInsight |
| http://localhost:9001 | MinIO console |

kafka-ui has no authentication at all. Do not put it behind Caddy.

---

## Step 12b — Trace the chain

`/stats` returning non-zero proves all six components at once. When it returns
zeros, check each link in turn.

### Producer → Kafka

kafka-ui at `localhost:8080` → Topics → `transactions` shows a count per
partition and the messages themselves. Or:

```bash
docker exec kafka kafka-run-class kafka.tools.GetOffsetShell \
  --bootstrap-server kafka:29092 --topic transactions
```

Per-partition write counts. Run it twice while the producer runs; climbing
numbers mean Kafka is receiving.

### Kafka → consumer

The most informative check in the system:

```bash
docker exec kafka kafka-consumer-groups --bootstrap-server kafka:29092 \
  --describe --group fraud-scorer
```

| Reading | Meaning |
|---|---|
| LAG ~0, offsets climbing | keeping up |
| LAG climbing | alive but too slow — `--scale fraud-consumer=N` |
| offsets frozen | stuck or dead |
| "no active members" | not running |

### Consumer → Redis

RedisInsight at `localhost:8001`, or:

```bash
R='redis-cli --no-auth-warning -a "$REDIS_PASSWORD"'
docker exec redis-stack sh -c "$R DBSIZE"
docker exec redis-stack sh -c "$R MGET stats:total_scored stats:total_fraud stats:high_risk"
docker exec redis-stack sh -c "$R ZREVRANGE flagged_transactions 0 4 WITHSCORES"
docker exec redis-stack sh -c "$R GET prediction:<TransactionID>"
```

`prediction:*` keys expire after an hour; `flagged_transactions` and `stats:*`
never do. An empty DBSIZE with a healthy consumer group means scoring is failing
— check `logs fraud-consumer`.

### Watch it run

```bash
# terminal 1
docker compose -f docker-compose.yml -f docker-compose.prod.yml \
  --profile load run --rm fraud-producer

# terminal 2
watch -n 2 'docker exec kafka kafka-consumer-groups \
  --bootstrap-server kafka:29092 --describe --group fraud-scorer'
```

---

## Step 13 — Day-2

```bash
# Shorthand for the rest of this section
alias dc='docker compose -f docker-compose.yml -f docker-compose.prod.yml'

dc logs -f fraud-consumer           # follow the scorer
dc ps                                # health
docker stats --no-stream             # actual memory vs the limits

dc up -d --scale fraud-consumer=6    # 6 partitions is the ceiling

# Deploy a change
git pull && dc up -d --build

# Back up the registry (Postgres + MinIO); volume prefix is the directory name
docker run --rm -v fraud-detect-system_mlflow-db:/v -v "$PWD":/b alpine \
  tar czf /b/mlflow-db-$(date +%F).tgz -C /v .
docker run --rm -v fraud-detect-system_minio-data:/v -v "$PWD":/b alpine \
  tar czf /b/minio-$(date +%F).tgz -C /v .

dc down                              # stop, keep volumes
dc down -v                           # DESTROYS every volume: registry, Kafka log, Redis
```

Hetzner's automatic backups (+20% of server cost) or a manual snapshot before
upgrades are worth it — the Postgres and MinIO volumes survive container
restarts, not host loss.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| No certificate, Caddy logs show ACME failure | DNS record missing or not propagated, or port 80 closed in the cloud firewall |
| `fraud-consumer` restarting, logs mention the alias | Registry empty — step 10 |
| API unhealthy for ~75s after start | Known: `kafka-python` retries bootstrap for ~45s and `max_block_ms` does not bound the constructor |
| `/stats` all zeros | Nothing produced yet, or the consumer is not scoring — check its logs |
| OOM kills | `docker stats`; limits sum to ~8.6 GB. Drop `--scale`, or resize the server |
| Grafana redirects to a wrong host | `GF_SERVER_ROOT_URL` — set from `DOMAIN` by the overlay, so check `DOMAIN` in `.env` |
| `docker compose` says a variable is missing | `.env` is incomplete; compose fails closed by design |

---

## What this deployment is not

Stated plainly, because a reviewer will ask:

- **Single broker, replication factor 1, Zookeeper mode.** Losing the broker
  loses undelivered messages. Real HA means managed Kafka.
- **Kafka and Redis are plaintext with no auth inside the Docker network.** The
  boundary is `BIND_ADDRESS=127.0.0.1` plus the cloud firewall.
- **One host.** No redundancy, no failover; restarts are visible downtime.
- **Ingestion is not idempotent.** Re-POSTing a `TransactionID` re-scores it and
  double-counts `stats:*`.
- **No alerting.** Prometheus collects, Grafana draws; nothing pages anyone.
