# mimir

Pure-Go machine learning framework (no cgo, no Python at runtime). Run
`go vet ./...` and `go test ./...` before pushing.

## Code tools

Two code-navigation tools are set up for Claude Code sessions in this repo.
Use graphify to find your way around and Serena to read and change code by
symbol.

Every task goes in this order:

1. Understand it first. Before grepping or opening files, run
   `graphify query` for the task, then use Serena's symbol tools (symbols
   overview, find symbol, find references) on what the graph points to.
2. Change code with Serena's symbol-level edits where they fit.
3. Keep both current. .claude/hooks/refresh-code-maps.sh rebuilds the graph
   (`graphify update .`) and Serena's index (`serena project index`) whenever
   the working tree has changed since the last refresh. The Stop hook runs it
   at the end of every turn and a PreToolUse hook runs it before each
   `git commit`; run it yourself when you need the fresh graph mid-turn.

### graphify

graphify builds a knowledge graph of the code in graphify-out/ (god nodes,
communities, cross-file relationships). The SessionStart hook rebuilds it, so
it is not committed.

- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code the hooks run `graphify update .` for you (AST only, no API cost); run .claude/hooks/refresh-code-maps.sh yourself if you need the updated graph before the turn ends.
- The full skill is in .claude/skills/graphify/ (`/graphify`).
- The graphify MCP server in .mcp.json serves the same graph as native tools
  (`query_graph`, `shortest_path`, `god_nodes`); it starts once
  graphify-out/graph.json exists.

### Serena

Serena is an MCP server (.mcp.json) that gives language-server tools: symbol
overviews, find symbol, find references, and symbol-level edits. Its own
instructions arrive with the server. Once graphify has pointed you at the
right file or symbol, read and edit it with Serena's symbolic tools rather
than whole-file reads. Project settings are in .serena/project.yml.

### Running these locally

Cloud sessions install everything in .claude/hooks/session-start.sh. On your
own machine, install once:

    uv tool install --python 3.13 serena-agent
    uv tool install 'graphifyy[sql]'
    go install golang.org/x/tools/gopls@latest

Without them the hooks do nothing, and uvx fetches Serena when the MCP
server first starts.
