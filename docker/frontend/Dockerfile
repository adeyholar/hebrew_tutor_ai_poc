FROM node:18-slim AS builder
WORKDIR /app
COPY src/frontend/package*.json ./
RUN npm ci --legacy-peer-deps
COPY src/frontend/ ./
RUN npm run build

FROM nginx:alpine
RUN adduser -D -g '' appuser
RUN mkdir -p /var/cache/nginx/client_temp /var/cache/nginx/proxy_temp /var/cache/nginx/fastcgi_temp /var/cache/nginx/uwsgi_temp /var/cache/nginx/scgi_temp /run/nginx
COPY docker/frontend/nginx.conf /etc/nginx/nginx.conf
COPY --from=builder /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]