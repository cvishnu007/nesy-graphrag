# Setup

Setup instructions now live in **[docs/RUN_FROM_SCRATCH.md](docs/RUN_FROM_SCRATCH.md)**. It covers:

- prerequisites and the Python environment (CUDA or CPU);
- `.env` configuration and starting Neo4j and Ollama;
- building every data store from scratch, in order;
- running the app;
- reproducing the paper's results;
- troubleshooting.

Before changing code or data, also read [AGENT_HANDOFF.md](AGENT_HANDOFF.md) for the safety rules. In particular, `NEO4J_ALLOW_RESET=true` clears the Neo4j database.

## Working on the repository

Start from an up-to-date `master` and work on a branch:

```powershell
git switch master
git pull --ff-only origin master
git switch -c your-branch-name
```
