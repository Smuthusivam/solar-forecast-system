# Multi-stage: build Vite bundle, then serve via nginx
FROM node:20-alpine AS builder

WORKDIR /app

# Install deps from lockfile for reproducible builds
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Build with VITE_API_BASE_URL baked in (empty default = same-origin via nginx proxy)
ARG VITE_API_BASE_URL=""
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

COPY frontend/ ./
RUN npm run build

# ─── Serve stage ───
FROM nginx:1.27-alpine AS runner

COPY --from=builder /app/dist /usr/share/nginx/html

# Store both configs in a templates dir; start.sh picks the right one at runtime
COPY frontend/nginx.conf     /etc/nginx/templates/nginx.conf
COPY frontend/nginx.dev.conf /etc/nginx/templates/nginx.dev.conf
COPY frontend/start.sh       /start.sh
RUN chmod +x /start.sh

EXPOSE 80 443

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- --max-redirect=1 http://127.0.0.1/ >/dev/null 2>&1 || wget -qO- http://127.0.0.1/assets/ >/dev/null 2>&1 || exit 1

CMD ["/start.sh"]
