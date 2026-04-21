#!/usr/bin/env sh
# Espera a que PostgreSQL acepte conexiones (útil en docker-compose).
set -e
host="${POSTGRES_HOST:-postgres}"
port="${POSTGRES_PORT:-5432}"
user="${POSTGRES_USER:-postgres}"
until pg_isready -h "$host" -p "$port" -U "$user" > /dev/null 2>&1; do
  echo "Esperando a PostgreSQL en ${host}:${port}..."
  sleep 2
done
echo "PostgreSQL listo."
