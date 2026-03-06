# MemForge MCP Server
# Multi-stage build: compile TypeScript, then run in slim Node image

# --- Build stage ---
FROM node:20-slim AS builder

WORKDIR /app

# Install deps first for layer caching
COPY package.json package-lock.json ./
RUN npm ci

# Copy source and compile
COPY tsconfig.json ./
COPY src/ ./src/
RUN npm run build

# --- Runtime stage ---
FROM node:20-slim

# curl is needed for Docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Production deps only
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# Compiled output from builder
COPY --from=builder /app/dist ./dist

ENV NODE_ENV=production
ENV MEMFORGE_PORT=3100

EXPOSE 3100

# The local embedding model (~30MB ONNX) downloads on first run
# and caches in /app/node_modules/@xenova/transformers/.cache
CMD ["node", "dist/server.js"]
