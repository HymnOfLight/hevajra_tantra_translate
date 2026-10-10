#!/usr/bin/env bash
# Connectivity check and cost projection on a machine with Claude access.
#
#   tools/server_check.sh [api|subscription] [--pilot]
#
# Steps (all in one run directory, runs/server-check-<UTC time>/):
#   1. fetch the source texts (skipped when data/raw/ already holds them)
#   2. ingest (segmentation and the G0 integrity check; no LLM call)
#   3. claude-check: one tiny live call (fractions of a cent, or a sliver of the plan allowance)
#   4. collate --dry-run: projected calls and tokens of a full collation; nothing is sent
#   5. only with --pilot: collate chapter I.1 once (about 1-3 real calls)
#
# It then packs the text-free results into server-check-<UTC time>.tar.gz in the repository
# root: the check summary, the cost projection, the LLM audit log (no prompt or answer text),
# the manifest and the G0 report, plus tool versions. Send that archive back; nothing in it
# contains licensed source text, API keys or login tokens.
#
# Backend "subscription" uses `claude -p` with the Claude Code login of the account running
# this script (run `claude` once and /login, or export CLAUDE_CODE_OAUTH_TOKEN from
# `claude setup-token`). Backend "api" needs ANTHROPIC_API_KEY in the environment.
set -euo pipefail

BACKEND="${1:-subscription}"
PILOT="${2:-}"
case "$BACKEND" in api|subscription) ;; *) echo "usage: $0 [api|subscription] [--pilot]" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="runs/server-check-$STAMP"
OUT="server-check-$STAMP"
HM=(hevajra-matrix --root "$ROOT" --run-dir "$RUN" --backend "$BACKEND")
mkdir -p "$RUN" "$OUT"

{
  echo "utc: $STAMP"
  echo "backend: $BACKEND"
  echo "git: $(git rev-parse HEAD 2>/dev/null || echo unknown) $(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  echo "python: $(python3 --version 2>&1)"
  echo "hevajra-matrix: $(python3 -c 'import hevajra_matrix; print(hevajra_matrix.__version__)' 2>&1)"
  echo "claude: $(claude --version 2>&1 || echo 'not installed')"
  echo "anthropic sdk: $(python3 -c 'import anthropic; print(anthropic.__version__)' 2>&1 | tail -n 1)"
  echo "api key set: $([ -n "${ANTHROPIC_API_KEY:-}" ] && echo yes || echo no)"
} > "$OUT/environment.txt"

step() { echo; echo "== $*"; }
status=0

step "fetch"
if ls data/raw/T18n0892.xml >/dev/null 2>&1; then echo "data/raw already holds the texts"; else hevajra-matrix --root "$ROOT" fetch; fi

step "ingest"
"${HM[@]}" ingest 2>&1 | tee "$OUT/ingest.log" || status=1

step "claude-check ($BACKEND)"
"${HM[@]}" claude-check 2>&1 | tee "$OUT/claude_check.log" || status=1

step "collate --dry-run"
"${HM[@]}" collate --dry-run 2>&1 | tee "$OUT/dry_run.log" || status=1

if [ "$PILOT" = "--pilot" ]; then
  step "pilot: collate I.1, one replicate"
  "${HM[@]}" collate --chapters I.1 --replicates 1 2>&1 | tee "$OUT/pilot.log" || status=1
fi

for f in claude_check.json collation/dry_run.json llm_audit.jsonl manifest.json ingest/g0.json; do
  [ -f "$RUN/$f" ] && cp "$RUN/$f" "$OUT/$(echo "$f" | tr '/' '_')"
done
tar -czf "$OUT.tar.gz" "$OUT"
rm -rf "$OUT"
echo
echo "packed: $ROOT/$OUT.tar.gz  (send this file back)"
echo "run directory kept on this machine: $ROOT/$RUN"
exit $status
