# The Home Guard API gateway. Build from the repo root:
#   docker build -f home_guard_project/gateway/Dockerfile -t homeguard-gateway .
# Only the gateway and box/providers.py (the provider and price tables) go into the image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HOMEGUARD_GATEWAY_CONFIG=/config/gateway.yaml HOMEGUARD_GATEWAY_DB=/data/gateway.sqlite3

WORKDIR /app
COPY home_guard_project/gateway/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY home_guard_project/box/__init__.py home_guard_project/box/providers.py /app/home_guard_project/box/
COPY home_guard_project/gateway/ /app/home_guard_project/gateway/

RUN useradd --system --uid 10001 gateway && mkdir -p /data /config && chown gateway /data
USER gateway
VOLUME ["/data"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4)"

ENTRYPOINT ["python", "-m", "home_guard_project.gateway"]
CMD ["serve"]
