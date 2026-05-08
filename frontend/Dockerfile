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
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD wget -qO- http://localhost/ >/dev/null || exit 1

CMD ["nginx", "-g", "daemon off;"]
