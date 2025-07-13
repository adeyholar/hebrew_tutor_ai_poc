# Builder stage (modular deps/TS build, scalable caching)
FROM node:20-alpine AS builder
WORKDIR /app
COPY src/frontend/package*.json src/frontend/tsconfig.json ./
RUN npm ci && npm audit fix
RUN npm install react-scripts -g --silent  # Global for PATH in alpine
COPY src/frontend /app
RUN npm run build

# Runtime stage (slim, secure serve)
FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/build /app/build
RUN npm install -g serve
CMD ["serve", "-s", "build", "-l", "3000"]