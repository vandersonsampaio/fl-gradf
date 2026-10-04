#!/usr/bin/env bash
# Janela de execução (regra do autor, 2026-10-04): NADA roda das 7h às 18h de segunda a sexta.
# Fora disso (noites e fins de semana) roda normalmente. A cada 30 s:
#   seg–sex e 7 ≤ hora < 18  -> SIGSTOP em todo o grupo de processos (lançador + xargs + runs), sem perder estado
#   caso contrário           -> SIGCONT
# Sai sozinho quando o grupo termina. Vale para qualquer grade lançada com setsid
# (o PGID é o PID do lançador, e os filhos herdam o grupo).
# Uso: bash scripts/janela_execucao.sh <PGID> <nome> [log]
set -u
PGID=${1:?PGID do lançador}
NOME=${2:?nome da grade}
LOG=${3:-results/janela_execucao.log}
estado=rodando
while pgrep -g "$PGID" >/dev/null 2>&1; do
  h=$((10#$(date +%H)))
  dow=$(date +%u)  # 1 = segunda ... 7 = domingo
  if [ "$dow" -le 5 ] && [ "$h" -ge 7 ] && [ "$h" -lt 18 ]; then
    if [ "$estado" = rodando ]; then
      kill -STOP -- "-$PGID" 2>/dev/null && estado=pausado && echo "$(date +%F\ %T) $NOME PAUSA (SIGSTOP grupo $PGID)" >> "$LOG"
    fi
  else
    if [ "$estado" = pausado ]; then
      kill -CONT -- "-$PGID" 2>/dev/null && estado=rodando && echo "$(date +%F\ %T) $NOME RETOMA (SIGCONT grupo $PGID)" >> "$LOG"
    fi
  fi
  sleep 30
done
echo "$(date +%F\ %T) $NOME FIM (grupo $PGID terminou)" >> "$LOG"
