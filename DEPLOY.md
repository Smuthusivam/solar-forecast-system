# Deployment Guide

Full-stack Docker deployment of the Solar Irradiance Forecasting System.
Three containers, one private network, one command to start everything.

## Architecture

```
                ┌────────────────────────────────────┐
                │        Docker network              │
                │        (solar-net, bridge)         │
   Browser ─►   │                                    │
   :80          │  ┌──────────┐                      │
                │  │ frontend │  nginx + React SPA   │
                │  │  :80     │  proxies /api/*      │
                │  └────┬─────┘                      │
                │       │                            │
                │       ▼                            │
                │  ┌──────────┐    ┌──────────────┐  │
                │  │ backend  │───►│  postgres    │  │
                │  │  :8000   │    │   :5432      │  │
                │  │ FastAPI  │    │  pg 16-alpine│  │
                │  │ + ML core│    └──────────────┘  │
                │  └──────────┘                      │
                └────────────────────────────────────┘
```

| Service    | Image base          | Port (host) | Purpose                                |
|------------|---------------------|-------------|----------------------------------------|
| `frontend` | nginx:1.27-alpine   | 80          | Serves built Vite SPA, proxies `/api/*`|
| `backend`  | python:3.11-slim    | 8000        | FastAPI + XGBoost/LightGBM/Prophet     |
| `postgres` | postgres:16-alpine  | 5432        | Persistent storage for runs & datasets |

## Prerequisites

- **Docker Desktop** (Windows/Mac) or **Docker Engine + Compose v2** (Linux) — tested with Docker 24+
- **8 GB RAM** recommended (ML training is memory-hungry)
- **Anthropic API key** — https://console.anthropic.com/

## Quick start

```bash
# 1. Configure secrets
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY and POSTGRES_PASSWORD

# 2. Build and start everything
docker compose up -d --build

# 3. Verify
docker compose ps
curl http://localhost/health
```

Then open **http://localhost** in a browser.

## Common operations

```bash
# View logs (all services)
docker compose logs -f

# Logs for one service
docker compose logs -f backend

# Restart one service after code change
docker compose up -d --build backend

# Stop everything (containers stay, volumes preserved)
docker compose stop

# Tear down (containers removed, volumes preserved)
docker compose down

# Tear down + wipe data (DESTRUCTIVE — drops Postgres + uploads)
docker compose down -v

# Open a shell inside the backend
docker compose exec backend bash

# Connect to Postgres from host
docker compose exec postgres psql -U solar -d solar_forecast
```

## Persistent volumes

These survive `docker compose down` and container rebuilds:

| Volume             | Mount path inside container       | Holds                          |
|--------------------|-----------------------------------|--------------------------------|
| `postgres_data`    | `/var/lib/postgresql/data`        | All Postgres tables/indexes    |
| `backend_uploads`  | `/app/backend/storage/uploads`    | Uploaded CSVs                  |
| `backend_exports`  | `/app/backend/storage/exports`    | Generated PDF/CSV exports      |
| `backend_models`   | `/app/ml_core/saved_models`       | Trained XGBoost/LightGBM files |

To wipe a single volume:

```bash
docker compose down
docker volume rm solar-forecast-system_backend_uploads
docker compose up -d
```

## Updating

After pulling new code:

```bash
docker compose up -d --build
```

The `--build` flag rebuilds only services whose source changed (Docker layer cache handles the rest).
The `init_db()` call on backend startup creates any new tables automatically — no manual migration needed for additive schema changes.

## Production checklist

Before exposing to the internet:

- [ ] Replace `POSTGRES_PASSWORD` with a strong random value
- [ ] Set `ALLOWED_ORIGINS` to your real domain (no `localhost`)
- [ ] Put a TLS-terminating reverse proxy (Caddy / Traefik / Cloudflare) in front of port 80
- [ ] Pin image versions in `docker-compose.yml` (already done for postgres + nginx)
- [ ] Set up off-site backups for the `postgres_data` volume:
      ```bash
      docker compose exec postgres pg_dump -U solar solar_forecast | gzip > backup-$(date +%F).sql.gz
      ```
- [ ] Bump backend workers in `backend/Dockerfile` CMD (`--workers 4` for 4-core hosts)
- [ ] Remove the host port mapping for `postgres` if external access isn't needed (delete the `ports:` block under `postgres`)

## Troubleshooting

**Backend exits with `ANTHROPIC_API_KEY must be set`**
The `?` syntax in compose enforces required vars. Set the key in `.env` and re-run.

**Frontend shows "Network Error" on every API call**
Check `docker compose logs frontend` for nginx errors. Most common cause: backend container is unhealthy — `docker compose ps` should show `backend` as `healthy`.

**Postgres "password authentication failed" after changing `POSTGRES_PASSWORD`**
The password is only set on first volume init. Either wipe the volume (`docker compose down -v`) or change it inside Postgres:
```bash
docker compose exec postgres psql -U solar -c "ALTER USER solar WITH PASSWORD 'newpass';"
```

**Out of memory during ML training**
Increase Docker Desktop memory (Settings → Resources → Memory ≥ 8 GB) or reduce model complexity.

**Build takes forever on first run**
Normal — Prophet/cmdstanpy compiles a Stan model (~5 min). Subsequent builds use the cache.

## Running services individually (without compose)

If you want to run only one service:

```bash
# Backend only (against an existing Postgres)
docker build -f backend/Dockerfile -t solar-backend .
docker run --rm -p 8000:8000 \
  -e DATABASE_URL=postgresql://solar:pass@host.docker.internal:5432/solar_forecast \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  solar-backend

# Frontend only (proxying to a remote backend)
docker build -f frontend/Dockerfile \
  --build-arg VITE_API_BASE_URL=https://api.example.com \
  -t solar-frontend .
docker run --rm -p 80:80 solar-frontend
```
