# mimir

Pure-Go machine learning framework (no cgo, no Python at runtime). Run
`go vet ./...` and `go test ./...` before pushing.

## Code navigation tools

This repo ships two MCP servers in `.mcp.json`, both launched with `uvx`
(install [uv](https://docs.astral.sh/uv/) once):

- **serena** ([oraios/serena](https://github.com/oraios/serena)): symbol-level
  navigation and editing through gopls. Needs `gopls` on `PATH`
  (`go install golang.org/x/tools/gopls@latest`). Configured in
  `.serena/project.yml`.
- **graphify** ([Graphify-Labs/graphify](https://github.com/Graphify-Labs/graphify)):
  a knowledge graph of the codebase in `graphify-out/` (gitignored). Build it
  once with `uvx --from graphifyy graphify extract . --code-only`, which
  parses the code locally with tree-sitter and needs no API key. For the
  `/graphify` skill and its search hooks, `uv tool install graphifyy` and run
  `graphify install --project`.

Run `/mcp` in Claude Code to check both servers are connected. The graphify
server only starts once `graphify-out/graph.json` exists.

How to use them:

- For a question about how parts of mimir connect (what calls what, what a
  change breaks), query the graph first: the graphify MCP tools, or
  `graphify query "<question>"`, `graphify path "<A>" "<B>"` and
  `graphify explain "<concept>"`. Read raw files only for lines the graph
  located. `graphify-out/GRAPH_REPORT.md` is for broad architecture review.
- For reading and editing Go code, prefer Serena's symbol tools
  (`get_symbols_overview`, `find_symbol`, `find_referencing_symbols`,
  `replace_symbol_body`) over whole-file reads.
- After changing code, run `graphify update .` to refresh the graph (AST
  only, no model calls).
