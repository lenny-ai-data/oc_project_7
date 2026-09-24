# Image de l'API RAG Puls-Events.
# Build : docker build -t puls-events .
# Run   : docker run --rm -p 8000:8000 --env-file .env puls-events

# Image python avec uv
FROM ghcr.io/astral-sh/uv:0.12.6-python3.13-trixie-slim

# Bytecode précompilé et copie des paquets
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Couche des dependances, seulement quand le lock change
# --no-default-groups : ni les outils dev ni l'interface de chat
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups

# Utilisateur non privilégié, créé avant les copies
RUN useradd --create-home --uid 1000 app

# Code et index retenu. data/ appartient à app car c'est le seul dossier écrit
COPY rag/ rag/
COPY api/ api/
COPY --chown=app:app data/index/chunk_1000/ data/index/chunk_1000/

USER app

# Port du conteneur
EXPOSE 8000

# exec pour que uvicorn devienne PID 1 et reçoive les signaux d'arrêt.
# PORT fourni par l'hébergeur ou 8000 en local.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
