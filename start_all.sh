#!/usr/bin/env bash
#
# start_all.sh — boot the full SOD stack in dependency order:
#   1. vLLM (MiniCPM-V)  → http://localhost:8010
#   2. FastAPI backend   → http://localhost:8000
#   3. Vite dev server   → http://localhost:5173 (or whatever Vite picks)
#
# Each service streams to logs/{vllm,backend,frontend}.log so this
# terminal stays readable. Ctrl+C tears everything down cleanly.
#
# Note on ports: the standalone vLLM command in the README uses --port 8000,
# but that collides with the FastAPI backend. We move vLLM to 8010 to match
# the backend's default VLM_API_BASE (app/core/config.py).
#
# Usage:
#   bash ~/start_all.sh               # full stack
#
# The script uses an absolute PROJECT_ROOT (set below) so it can live
# anywhere — drop it in $HOME for one-command boot at login.
#
# Pre-existing vLLM on :8010 is reused (and left running on Ctrl+C).
# ----------------------------------------------------------------------------

set -euo pipefail
set -m   # job control: each `&` lands in its own process group → clean kill

# ── Paths ────────────────────────────────────────────────────────────────
# Absolute project root so this script works wherever it lives
# (e.g. ~/start_all.sh after copying it to the home directory).
PROJECT_ROOT="/home/user/Small_object_detection/sodv3/SOD"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

VLLM_VENV="/home/user/Small_object_detection/local-deploy/.venv-minicpm/bin/activate"
VLLM_MODEL="/home/user/Small_object_detection/local-deploy/models/MiniCPM-V-4_5"
VLLM_PORT=8010
BACKEND_PORT=8000

# Backend lives in a conda env (loguru, sqlalchemy, asyncpg, … all installed
# there — base Python 3.13 doesn't have them).
CONDA_SH="/home/user/home/enter/etc/profile.d/conda.sh"
BACKEND_CONDA_ENV="sodv2"

VLLM_LOG="$LOG_DIR/vllm.log"
BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"

# ── ANSI helpers ─────────────────────────────────────────────────────────
c_b=$'\033[1m'; c_dim=$'\033[2m'; c_r=$'\033[31m'; c_g=$'\033[32m'
c_y=$'\033[33m'; c_blu=$'\033[34m'; c_e=$'\033[0m'
say()  { printf '\n%s %s%s%s\n' "${c_blu}▶${c_e}" "$c_b" "$*" "$c_e"; }
ok()   { printf '%s %s\n'        "${c_g}✓${c_e}" "$*"; }
warn() { printf '%s %s\n'        "${c_y}!${c_e}" "$*"; }
die()  { printf '%s %s\n' "${c_r}✗${c_e}" "$*" >&2; exit 1; }

# ── Sanity checks ────────────────────────────────────────────────────────
[[ -f "$VLLM_VENV"  ]] || die "vLLM venv not found: $VLLM_VENV"
[[ -d "$VLLM_MODEL" ]] || die "vLLM model dir not found: $VLLM_MODEL"
[[ -f "$BACKEND_DIR/start.sh" ]] || die "backend/start.sh missing"
[[ -d "$FRONTEND_DIR/node_modules" ]] \
  || die "frontend/node_modules missing — run 'cd frontend && npm install' first"
[[ -f "$CONDA_SH" ]] || die "conda.sh not found: $CONDA_SH"
[[ -d "$(dirname "$CONDA_SH")/../../envs/$BACKEND_CONDA_ENV" ]] \
  || die "conda env '$BACKEND_CONDA_ENV' not found"

command -v curl >/dev/null || die "curl is required"
command -v ss   >/dev/null || die "iproute2 ss(8) is required"

port_busy() { ss -ltn "sport = :$1" 2>/dev/null | tail -n +2 | grep -q .; }
port_busy "$BACKEND_PORT" && die "Port $BACKEND_PORT already in use (backend)"

# ── Process tracking + cleanup ──────────────────────────────────────────
VLLM_PID=""; BACKEND_PID=""; FRONTEND_PID=""
VLLM_REUSE=0
DONE=0

cleanup() {
  (( DONE )) && return
  DONE=1
  trap '' INT TERM EXIT
  echo
  say "Shutting down…"
  # Reverse order: frontend first, then backend, then vLLM.
  for pid in "$FRONTEND_PID" "$BACKEND_PID"; do
    [[ -n "$pid" ]] && kill -TERM -- -"$pid" 2>/dev/null || true
  done
  if (( !VLLM_REUSE )); then
    [[ -n "$VLLM_PID" ]] && kill -TERM -- -"$VLLM_PID" 2>/dev/null || true
  else
    warn "Leaving pre-existing vLLM running on :$VLLM_PORT"
  fi
  sleep 2
  for pid in "$FRONTEND_PID" "$BACKEND_PID" "$VLLM_PID"; do
    [[ -n "$pid" ]] && kill -KILL -- -"$pid" 2>/dev/null || true
  done
  ok "All managed services stopped."
}
trap cleanup INT TERM EXIT

# ── 1. vLLM ─────────────────────────────────────────────────────────────
if curl -fs "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
  warn "Reusing vLLM already serving on :$VLLM_PORT — skipping launch"
  VLLM_REUSE=1
elif port_busy "$VLLM_PORT"; then
  die "Port $VLLM_PORT in use by something other than vLLM"
else
  say "Launching vLLM on :$VLLM_PORT (logs → $VLLM_LOG)"
  : > "$VLLM_LOG"

  bash -c "
    source '$VLLM_VENV'
    export VLLM_USE_V1=0
    export CUDA_VISIBLE_DEVICES=0,1
    exec vllm serve '$VLLM_MODEL' \
      --served-model-name MiniCPM-V-4_5 \
      --host 0.0.0.0 \
      --port $VLLM_PORT \
      --dtype auto \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.90 \
      --max-num-seqs 1 \
      --tensor-parallel-size 2 \
      --trust-remote-code \
      --tokenizer-mode slow
  " >>"$VLLM_LOG" 2>&1 &
  VLLM_PID=$!

  say "Waiting for vLLM /v1/models — first-time model load can take 3–10 min…"
  for i in $(seq 1 600); do
    if curl -fs "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
      ok "vLLM ready after ${i}s"
      break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      die "vLLM crashed — last lines of $VLLM_LOG:
$(tail -n 30 "$VLLM_LOG" 2>/dev/null)"
    fi
    if (( i % 30 == 0 )); then
      printf '   %s%ds elapsed — last log line:%s\n' "$c_dim" "$i" "$c_e"
      tail -n 1 "$VLLM_LOG" 2>/dev/null | sed 's/^/     /'
    fi
    sleep 1
    [[ $i -eq 600 ]] && die "vLLM did not become ready in 10 min — see $VLLM_LOG"
  done
fi

# ── 2. Backend ──────────────────────────────────────────────────────────
say "Launching backend on :$BACKEND_PORT (logs → $BACKEND_LOG)"
: > "$BACKEND_LOG"
bash -c "
  source '$CONDA_SH'
  conda activate '$BACKEND_CONDA_ENV'
  exec bash '$BACKEND_DIR/start.sh'
" >>"$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

# Poll /openapi.json instead of a fixed sleep — ensures uvicorn really
# accepted a request before we hit it from the frontend.
for i in $(seq 1 30); do
  if curl -fs "http://localhost:$BACKEND_PORT/openapi.json" >/dev/null 2>&1; then
    ok "Backend ready after ${i}s"
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    die "backend died during startup — last lines of $BACKEND_LOG:
$(tail -n 30 "$BACKEND_LOG" 2>/dev/null)"
  fi
  sleep 1
  [[ $i -eq 30 ]] && die "backend not ready in 30s — see $BACKEND_LOG"
done

# ── 3. Frontend ─────────────────────────────────────────────────────────
say "Launching frontend (Vite) — logs → $FRONTEND_LOG"
: > "$FRONTEND_LOG"
bash -c "cd '$FRONTEND_DIR' && npm run dev" >>"$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!

url=""
for _ in $(seq 1 30); do
  url=$(grep -oE 'http://localhost:[0-9]+' "$FRONTEND_LOG" 2>/dev/null | head -1 || true)
  [[ -n "$url" ]] && break
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    die "frontend died — last lines of $FRONTEND_LOG:
$(tail -n 30 "$FRONTEND_LOG" 2>/dev/null)"
  fi
  sleep 1
done
[[ -n "$url" ]] && ok "Frontend ready at $url" \
                || warn "Frontend still warming up — tail $FRONTEND_LOG"

# ── Status banner ───────────────────────────────────────────────────────
cat <<EOF

${c_b}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${c_e}
${c_b}All services running${c_e}
  vLLM     : http://localhost:$VLLM_PORT/v1
  Backend  : http://localhost:$BACKEND_PORT/docs
  Frontend : ${url:-(see $FRONTEND_LOG)}

Tail any log:
  tail -f $VLLM_LOG
  tail -f $BACKEND_LOG
  tail -f $FRONTEND_LOG

Press Ctrl+C to stop everything.
${c_b}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${c_e}
EOF

# ── Block until any service exits (or user Ctrl+Cs) ─────────────────────
PIDS=()
(( VLLM_REUSE )) || PIDS+=("$VLLM_PID")
PIDS+=("$BACKEND_PID" "$FRONTEND_PID")

wait -n "${PIDS[@]}" 2>/dev/null || true
warn "A service exited — bringing the rest down"
# EXIT trap → cleanup() handles the rest.
