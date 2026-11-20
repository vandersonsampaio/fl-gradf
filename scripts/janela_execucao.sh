#!/usr/bin/env bash
# Execution window: keeps a grid from running 8h–18h Monday to Friday (working hours);
# outside that (nights and weekends) it runs normally. Every 30 s:
#   Mon–Fri and 8 ≤ hour < 18  -> SIGSTOP on the whole process group (launcher + xargs + runs), without losing state
#   otherwise                  -> SIGCONT
# Exits on its own when the group ends. Works for any grid launched with setsid
# (the PGID is the launcher's PID, and the children inherit the group).
# Optional one-off exception, authorized by the author: between FREE_FROM and
# FREE_UNTIL (any `date -d` string) the group is never paused, even inside the
# window.
# Usage: [FREE_FROM="..." FREE_UNTIL="..."] bash scripts/janela_execucao.sh <PGID> <name> [log]
set -u
PGID=${1:?launcher PGID}
NOME=${2:?grid name}
LOG=${3:-results/janela_execucao.log}
FREE_FROM_S=$( [ -n "${FREE_FROM:-}" ] && date -d "$FREE_FROM" +%s || echo 0 )
FREE_UNTIL_S=$( [ -n "${FREE_UNTIL:-}" ] && date -d "$FREE_UNTIL" +%s || echo 0 )
[ "$FREE_UNTIL_S" -gt 0 ] && echo "$(date +%F\ %T) $NOME controller: no pause from $(date -d @"$FREE_FROM_S" +%F\ %T) to $(date -d @"$FREE_UNTIL_S" +%F\ %T) (author's exception)" >> "$LOG"
estado=rodando
while pgrep -g "$PGID" >/dev/null 2>&1; do
  h=$((10#$(date +%H)))
  dow=$(date +%u)  # 1 = Monday ... 7 = Sunday
  now=$(date +%s)
  if [ "$now" -ge "$FREE_FROM_S" ] && [ "$now" -lt "$FREE_UNTIL_S" ]; then
    inwindow=no
  elif [ "$dow" -le 5 ] && [ "$h" -ge 8 ] && [ "$h" -lt 18 ]; then
    inwindow=yes
  else
    inwindow=no
  fi
  if [ "$inwindow" = yes ]; then
    if [ "$estado" = rodando ]; then
      kill -STOP -- "-$PGID" 2>/dev/null && estado=pausado && echo "$(date +%F\ %T) $NOME PAUSE (SIGSTOP group $PGID)" >> "$LOG"
    fi
  else
    if [ "$estado" = pausado ]; then
      kill -CONT -- "-$PGID" 2>/dev/null && estado=rodando && echo "$(date +%F\ %T) $NOME RESUME (SIGCONT group $PGID)" >> "$LOG"
    fi
  fi
  sleep 30
done
echo "$(date +%F\ %T) $NOME END (group $PGID finished)" >> "$LOG"
