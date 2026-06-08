---
inclusion: always
---

# Graphify — Mandatory Context

**BEFORE answering any question about this codebase**, you MUST:

1. Read `graphify-out/GRAPH_REPORT.md` to understand god nodes, communities, module boundaries, and surprising connections.
2. Use the graph for navigation — prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over raw grep/glob when exploring architecture or dependencies.
3. When modifying code, consult the graph to understand what depends on the file you're changing.

This ensures every response is architecturally informed. The graph is the source of truth for structure; raw code is the source of truth for implementation details.
