# Code tools

Two code-navigation tools are set up for Claude Code sessions in this repo.
Use graphify to find your way around and Serena to read and change code by
symbol.

## graphify

graphify builds a knowledge graph of the code in graphify-out/ (god nodes,
communities, cross-file relationships). The SessionStart hook rebuilds it, so
it is not committed.

- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST only, no API cost).
- The full skill is in .claude/skills/graphify/ (`/graphify`).

## Serena

Serena is an MCP server (.mcp.json) that gives language-server tools: symbol
overviews, find symbol, find references, and symbol-level edits. Its own
instructions arrive with the server. Once graphify has pointed you at the
right file or symbol, read and edit it with Serena's symbolic tools rather
than whole-file reads. Project settings are in .serena/project.yml.

## Running these locally

Cloud sessions install everything in .claude/hooks/session-start.sh. On your
own machine, install once:

    uv tool install --python 3.13 serena-agent
    uv tool install 'graphifyy[sql]'
    go install golang.org/x/tools/gopls@latest

Without them the hooks do nothing, and uvx fetches Serena when the MCP
server first starts.
