#!/bin/bash
# Keeps the graphify code graph and Serena's symbol index current with the code.
# Runs from the Stop hook (end of each Claude turn) and, with --before-commit,
# from a PreToolUse hook on Bash, where it acts only on `git commit`. It does
# nothing when the working tree has not changed since the last refresh, so it
# is cheap to call often. --mark only records the current state, for the
# SessionStart hook, which builds both itself. Output goes to stderr to stay
# out of Claude's context.
set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}" || exit 0
export PATH="$HOME/.local/bin:$PATH"

# As a PreToolUse hook the tool call arrives as JSON on stdin.
if [ "${1:-}" = "--before-commit" ]; then
  grep -qE 'git( -C [^ ]+)? commit' || exit 0
fi

stamp=graphify-out/.code-maps-stamp

# A hash of HEAD, tracked changes and untracked files. graphify-out/ and
# Serena's cache are gitignored, so a refresh does not change it.
fingerprint() {
  {
    git rev-parse HEAD
    git diff HEAD
    git ls-files -o --exclude-standard -z | xargs -0 -r git hash-object --
  } 2>/dev/null | git hash-object --stdin
}

now=$(fingerprint)
if [ "${1:-}" = "--mark" ]; then
  mkdir -p graphify-out
  echo "$now" >"$stamp"
  exit 0
fi
if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$now" ]; then
  exit 0
fi

ok=true
if command -v graphify >/dev/null; then
  graphify update . >/dev/null 2>&1 || { echo "graphify update failed; the code graph may be stale" >&2; ok=false; }
fi
if command -v serena >/dev/null; then
  serena project index --log-level ERROR . >/dev/null 2>&1 || { echo "serena project index failed; Serena's symbol cache may be stale" >&2; ok=false; }
fi

if $ok; then
  mkdir -p graphify-out
  echo "$now" >"$stamp"
fi
exit 0
