# Home Guard API gateway, Railway build (context = this folder only).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HOMEGUARD_GATEWAY_CONFIG=/app/gateway.yaml HOMEGUARD_GATEWAY_DB=/data/gateway.sqlite3
WORKDIR /app
COPY home_guard_project/gateway/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY home_guard_project/ /app/home_guard_project/
COPY gateway.yaml /app/gateway.yaml
RUN mkdir -p /data
EXPOSE 8080
CMD ["python", "-m", "home_guard_project.gateway", "serve", "--host", "0.0.0.0", "--port", "8080"]
