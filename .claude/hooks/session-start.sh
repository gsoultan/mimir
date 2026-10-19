#!/bin/bash
# Gets Serena and Graphify ready for a Claude Code session and refreshes the
# code graph. Installs only run in Claude Code cloud sessions; locally, install
# the tools yourself (see CLAUDE.md). Output goes to stderr because a
# SessionStart hook's stdout is added to Claude's context.
set -euo pipefail
cd "$CLAUDE_PROJECT_DIR"
export PATH="$HOME/.local/bin:$PATH"

if [ "${CLAUDE_CODE_REMOTE:-}" = "true" ]; then
  {
    command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    command -v serena >/dev/null || uv tool install --python 3.13 serena-agent
    command -v graphify >/dev/null || uv tool install 'graphifyy[sql]'
    # Serena's Go support needs gopls. Put it next to the uv tools so the
    # Serena MCP server, which gets ~/.local/bin on PATH from .mcp.json, finds it.
    command -v gopls >/dev/null || GOBIN="$HOME/.local/bin" go install golang.org/x/tools/gopls@latest
  } >&2
  echo "export PATH=\"$HOME/.local/bin:\$PATH\"" >> "${CLAUDE_ENV_FILE:-/dev/null}"
fi

# Rebuild graphify-out/ from source. AST only: no API key, a few seconds.
if command -v graphify >/dev/null; then
  graphify update . >&2 || echo "graphify update failed; the code graph may be stale" >&2
fi

# Warm Serena's symbol cache in the background so startup is not held up, and
# record the tree state so the Stop hook only refreshes after code changes.
if command -v serena >/dev/null; then
  nohup serena project index --log-level ERROR . </dev/null >/dev/null 2>&1 &
fi
"$CLAUDE_PROJECT_DIR"/.claude/hooks/refresh-code-maps.sh --mark

# Stdout from this hook is added to Claude's context.
echo "Code tools: run \`graphify query\` and use Serena's symbol tools to understand the task before grepping or opening files. The graph and Serena's index refresh automatically after code changes."
