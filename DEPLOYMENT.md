# Deployment Guide

This document covers running the Deepfake Forensics Engine outside of a
developer venv: via Docker for local demos / academic defense, and notes
on what changes for cloud / GPU deployments.

## TL;DR — local demo in three commands

```bash
git clone https://github.com/Americo05/DFForensics.git
cd DFForensics
docker compose up --build
```

Then open <http://localhost:3000>. The first build downloads PyTorch
(CPU-only, ~200 MB), the dima806 ViT (~340 MB) on first request, and the
MesoNet weights (~160 KB, baked into the engine image during build).
Total build time on a fresh machine: ~5 min on broadband, ~15 min on
slow connections.

## What gets deployed

The `docker compose up` command starts two services:

| Service    | Image                  | Port | What it does                                  |
|------------|------------------------|------|-----------------------------------------------|
| `engine`   | `deepfake-engine:local`   | 8000 | FastAPI backend, plugin pipeline, SSE stream  |
| `frontend` | `deepfake-frontend:local` | 3000 | Next.js dashboard, served standalone          |

The frontend's `getApiBase()` builds the API URL from
`window.location.hostname:8000` at run time, so as long as the user
loads the dashboard from the same host that runs the engine, no extra
configuration is needed.

## Prerequisites

- **Docker Engine 24+** with the `compose` plugin (Docker Desktop on
  Windows / macOS bundles both).
- **~5 GB free disk** for the two images plus the Hugging Face cache the
  ViT plugin will create on first run.
- **No NVIDIA driver required** — the images are CPU-only by default.

## Configuration

All configuration is done via environment variables. Copy the template:

```bash
cp .env.example .env
# edit .env to set ENGINE_API_KEYS, SIGHTENGINE_*, etc.
```

Then `docker compose up` will pick up `.env` automatically.

| Variable                    | Default  | Purpose                                            |
|-----------------------------|----------|----------------------------------------------------|
| `ENGINE_API_KEYS`           | empty    | Comma-separated valid X-API-Key values. Empty = no auth. |
| `ENGINE_RATE_LIMIT_PER_MIN` | `10`     | Sliding-window rate limit per key/IP.              |
| `ENGINE_MAX_UPLOAD_MB`      | `200`    | Cap on uploaded media size in MB.                  |
| `SIGHTENGINE_ENABLED`       | `false`  | Whether the cloud detector runs.                   |
| `SIGHTENGINE_API_USER`      | empty    | Sightengine credentials (see https://sightengine.com). |
| `SIGHTENGINE_API_SECRET`    | empty    | Same.                                              |
| `NEXT_PUBLIC_ENGINE_API_KEY`| empty    | Frontend-side API key. Must match one in `ENGINE_API_KEYS`. |

> **Important**: `NEXT_PUBLIC_*` variables are inlined into the
> JavaScript bundle at build time. Changing them requires
> `docker compose build frontend` — `restart` is not enough.

## Common operations

### View logs

```bash
docker compose logs -f engine    # follow the backend
docker compose logs -f frontend  # follow the frontend
docker compose logs -f           # both interleaved
```

### Restart after code changes

Rebuild only what changed:

```bash
docker compose up --build engine     # backend change
docker compose up --build frontend   # frontend change
```

### Stop everything

```bash
docker compose down             # stop + remove containers
docker compose down --volumes   # also wipe healthcheck cache, etc.
```

### Check engine health

The engine exposes `/health` with per-plugin status:

```bash
curl http://localhost:8000/health | jq
```

Docker Compose also runs this every 30 s; `docker compose ps` will mark
the engine `(healthy)` once the models finish loading (~60 s cold start).

## Re-using a downloaded model cache between builds

Hugging Face caches models under `/root/.cache/huggingface` inside the
container by default. On a fresh build that cache is empty, so the
~340 MB ViT downloads again. To persist it across rebuilds:

```yaml
# docker-compose.yml — under services.engine
services:
  engine:
    volumes:
      - hf-cache:/root/.cache/huggingface

volumes:
  hf-cache:
```

Add the two snippets above to your local `docker-compose.yml` (not
committed — they're an optimisation, not a correctness fix).

## Enabling GPU acceleration (NVIDIA)

The default image installs the **CPU-only** PyTorch wheels because they
work on every machine. Inference is meaningfully slower (~10× for the
ViT plugin) but the pipeline is still usable for demos.

To run with CUDA acceleration on a host with an NVIDIA GPU + driver:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
   so Docker can expose `/dev/nvidia*` into containers.

2. Edit `Dockerfile.engine`. Replace the CPU PyTorch install:

   ```dockerfile
   RUN pip install --no-cache-dir \
       torch==2.5.1+cpu \
       torchvision==0.20.1+cpu \
       --index-url https://download.pytorch.org/whl/cpu
   ```

   With the CUDA build matching your driver (CUDA 12.x → cu124):

   ```dockerfile
   RUN pip install --no-cache-dir \
       torch==2.5.1 \
       torchvision==0.20.1 \
       --index-url https://download.pytorch.org/whl/cu124
   ```

   And add an NVIDIA base image:

   ```dockerfile
   FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime
   ```

3. Edit `docker-compose.yml` — under `services.engine`:

   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: 1
             capabilities: [gpu]
   ```

4. Rebuild and verify GPU is detected:

   ```bash
   docker compose up --build engine
   docker compose exec engine python -c "import torch; print(torch.cuda.is_available())"
   # expect: True
   ```

The image grows from ~3 GB to ~6 GB; that's the NVIDIA runtime + CUDA
libraries. Worth it for batch benchmarking, overkill for the dashboard.

## AMD GPU (ROCm)

Technically possible via `rocm/pytorch` base images, but:

- **Linux only** — ROCm has no Windows support beyond limited WSL2 hacks.
- **Restricted hardware** — only RX 6800/6900/7900, MI series, Radeon Pro.
- **~8 GB image** vs 3 GB CPU-only.

Treated as future work; not actively supported. If you have a compatible
RDNA2+ card, the changes mirror the NVIDIA section: swap base image,
swap pip wheels (`--index-url https://download.pytorch.org/whl/rocm6.0`),
pass the right `/dev/kfd` + `/dev/dri` devices to the container.

## Troubleshooting

### Engine container restarts in a loop / unhealthy

Check the logs:

```bash
docker compose logs --tail=200 engine
```

Most likely causes:

| Symptom                                              | Cause                                            | Fix                                                                                  |
|------------------------------------------------------|--------------------------------------------------|--------------------------------------------------------------------------------------|
| `OSError: ... offline mode is enabled`               | First-run ViT download blocked                   | Confirm internet inside the container. Or pre-populate the HF cache via the volume trick above. |
| `AttributeError: module 'mediapipe' has no attribute 'solutions'` | mediapipe got upgraded > 0.10.21                 | Pin it back in `requirements.txt` (already pinned to `0.10.14`).                     |
| `RuntimeError: CUDA out of memory`                   | Only on GPU build with small VRAM                | Reduce batch size, or fall back to the CPU image.                                    |
| Healthcheck times out after 60 s                     | Slow disk or first-time HF model download        | Increase `start_period` in `docker-compose.yml`.                                     |

### Frontend can't reach the engine

`getApiBase()` resolves the engine URL from `window.location.hostname`.
For local browsers that's `localhost`, which maps cleanly. If you deploy
the engine on a different host than the frontend, the current frontend
won't follow — pass the URL explicitly:

1. Add to your `.env`:

   ```bash
   NEXT_PUBLIC_ENGINE_API_BASE=https://your-engine-host.example.com
   ```

2. Edit `src/app/page.tsx` (and `src/components/VideoForensicsPlayer.tsx`)
   to consult that variable first:

   ```typescript
   function getApiBase(): string {
     const override = process.env.NEXT_PUBLIC_ENGINE_API_BASE;
     if (override) return override;
     if (typeof window === "undefined") return "http://localhost:8000";
     const host = window.location.hostname === "localhost"
       ? "127.0.0.1"
       : window.location.hostname;
     return `http://${host}:8000`;
   }
   ```

3. Rebuild the frontend (`NEXT_PUBLIC_*` is build-time, not runtime).

### Image build fails on Windows with "no space left on device"

Docker Desktop on Windows defaults to a 64 GB virtual disk. The two
images plus build cache exceed that easily. Either:

- Docker Desktop → Settings → Resources → expand the disk image.
- Periodically: `docker system prune --all --volumes` (warning:
  destructive — wipes everything not currently running).

## Production hardening checklist

The shipped configuration is tuned for local demo, not a public
production endpoint. Before exposing this to the internet:

- [ ] **Set `ENGINE_API_KEYS`** — without it the API is open.
- [ ] **Put a reverse proxy** (nginx, Caddy, Traefik) with TLS in
      front of both services. Don't expose 8000/3000 directly.
- [ ] **Set `ENGINE_RATE_LIMIT_PER_MIN` to a stricter value** for the
      public side (e.g. `3` per key).
- [ ] **Reduce `ENGINE_MAX_UPLOAD_MB`** to the smallest size your
      use case actually needs.
- [ ] **Rotate the HF cache and Sightengine credentials regularly**.
- [ ] **Add monitoring** — at minimum, hit `/health` from a curl loop
      and alert on non-200 responses for >5 minutes.
- [ ] **Restrict CORS origins** in `engine/main.py` to the dashboard's
      actual hostname (currently uses a permissive list).
