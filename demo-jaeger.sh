#!/usr/bin/env bash
#
# Sobe o que a demonstração precisa (pgvector + Jaeger), aponta os traces para o
# Jaeger e abre o chat. Nada é gravado no .env: as variáveis valem só para esta
# execução.
#
#   ./demo-jaeger.sh            # sobe tudo e abre o chat
#   ./demo-jaeger.sh --reset    # o mesmo, com a lista de traces do Jaeger zerada
#   ./demo-jaeger.sh --cleanup  # remove o pgvector avulso criado por este script
#
set -euo pipefail

cd "$(dirname "$0")"

COMPOSE_PG=fc-rag-pgvector       # o pgvector do docker-compose, na 5433
STANDALONE_PG=fc-rag-pgvector-demo
JAEGER=fc-jaeger
JAEGER_UI=http://localhost:16686
OTLP_ENDPOINT=http://localhost:4318
QUESTION='Qual é o SLA para incidentes P1 no plano Enterprise?'

reset_jaeger=false
case "${1:-}" in
  --reset)   reset_jaeger=true; shift ;;
  --cleanup) docker rm -f "$STANDALONE_PG" >/dev/null 2>&1 || true
             echo "Removido: $STANDALONE_PG (o volume com o índice fica)."
             exit 0 ;;
  "")        ;;
  *)         echo "Uso: $0 [--reset|--cleanup]" >&2; exit 2 ;;
esac

running() { [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = true ]; }
exists()  { docker inspect "$1" >/dev/null 2>&1; }

taken()   { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

free_port() {
  for port in "$@"; do
    if ! taken "$port"; then
      echo "$port"
      return 0
    fi
  done
  echo "Nenhuma porta livre entre: $*" >&2
  return 1
}

# --- Postgres -----------------------------------------------------------------
# Se o container do compose já está no ar, é ele. Senão, um avulso numa porta
# livre, sobre o mesmo volume — assim a 5433, se estiver ocupada por outro
# projeto, não é disputada.
if running "$COMPOSE_PG"; then
  pg_port=5433
  echo "Postgres: $COMPOSE_PG (porta $pg_port)."
elif running "$STANDALONE_PG"; then
  pg_port=$(docker port "$STANDALONE_PG" 5432/tcp | head -1 | sed 's/.*://')
  echo "Postgres: $STANDALONE_PG (porta $pg_port)."
elif exists "$COMPOSE_PG" && ! taken 5433; then
  docker start "$COMPOSE_PG" >/dev/null
  pg_port=5433
  echo "Postgres: $COMPOSE_PG reiniciado (porta $pg_port)."
elif exists "$STANDALONE_PG"; then
  docker start "$STANDALONE_PG" >/dev/null
  pg_port=$(docker port "$STANDALONE_PG" 5432/tcp | head -1 | sed 's/.*://')
  echo "Postgres: $STANDALONE_PG reiniciado (porta $pg_port)."
else
  pg_port=$(free_port 5433 5434 5435 5436)
  docker run -d --name "$STANDALONE_PG" \
    -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=rag_db \
    -p "$pg_port":5432 -v rag_postgres_data:/var/lib/postgresql/data \
    pgvector/pgvector:pg17 >/dev/null
  echo "Postgres: $STANDALONE_PG criado (porta $pg_port)."
fi

printf 'Esperando o Postgres'
for _ in $(seq 30); do
  if docker exec "$STANDALONE_PG" pg_isready -U postgres -d rag_db >/dev/null 2>&1 \
     || docker exec "$COMPOSE_PG" pg_isready -U postgres -d rag_db >/dev/null 2>&1; then
    echo " ok."
    break
  fi
  printf '.'
  sleep 1
done

# --- Jaeger -------------------------------------------------------------------
if ! running "$JAEGER"; then
  docker compose --profile jaeger up -d jaeger >/dev/null 2>&1
  echo "Jaeger: subido."
elif [ "$reset_jaeger" = true ]; then
  docker restart "$JAEGER" >/dev/null
  echo "Jaeger: reiniciado (lista de traces vazia)."
else
  echo "Jaeger: já no ar."
fi

printf 'Esperando o Jaeger'
for _ in $(seq 30); do
  if curl -sf "$JAEGER_UI/api/services" >/dev/null 2>&1; then
    echo " ok."
    break
  fi
  printf '.'
  sleep 1
done

# --- Ambiente da execução -----------------------------------------------------
# development para o sample rate ser 1.0 (todo trace é gravado) e as chaves do
# Langfuse vazias para o exportador usar o endpoint do Jaeger.
export DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:$pg_port/rag_db"
export APP_ENV=development
export OBSERVABILITY_ENABLED=true
export OTEL_TRACES_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT="$OTLP_ENDPOINT"
export LANGFUSE_PUBLIC_KEY=
export LANGFUSE_SECRET_KEY=

source .venv/bin/activate

cat <<EOF

--------------------------------------------------------------------
Pergunta para colar no chat:

  $QUESTION

Depois digite 'exit' e abra $JAEGER_UI
(serviço fcai-rag-api → Find Traces).
--------------------------------------------------------------------

EOF

exec python -m app.rag_chat
