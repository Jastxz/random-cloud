---
inclusion: always
---

# Skills Detector — Mandatory Stack Analysis

**When setting up this project, adding dependencies, or making architectural decisions**, you MUST:

1. Run `node /home/javier/Repos/skills-detector/dist/cli.js --cwd .` to detect the current stack (languages, frameworks, tools, testing).
2. If the output recommends curated skills (e.g., `vercel-labs/next-skills`, `vercel-labs/agent-skills`), review and follow those skill guidelines for best practices.
3. If `skills.json` exists in this repo, read it to understand which skills apply to this project and follow their recommendations.

## When to Run
- First interaction with this project in a new session
- After adding/removing major dependencies
- Before recommending architectural patterns or libraries

## Usage
```bash
# Full analysis with skill recommendations
node /home/javier/Repos/skills-detector/dist/cli.js --cwd .

# Quick detection only (no search)
node /home/javier/Repos/skills-detector/dist/cli.js --json --skip-search --cwd .
```

The detected stack informs which coding standards, patterns, and tools are appropriate for this project. Never recommend tools or patterns that conflict with the detected stack.
