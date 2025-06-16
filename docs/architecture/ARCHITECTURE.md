# Agentic Researcher System – Architecture (2025-06)

> This document supersedes any previous *architecture* files that referred to Docker
> containers, `config.toml`, or sandboxed execution.  The project now uses a single
> **`config.yaml`** file and executes user code directly on the host OS through the
> `TerminalRunner` service.

---

## 1 · Macro-view

```
┌─────────────────────────────────────────┐
│             Front-End (Svelte)          │
│  ─ chat UI  ─ code editor ─ terminal ─  │ 
└─────────────────────────────────────────┘
               ▲               │ Socket.IO / HTTP
               │               ▼
┌─────────────────────────────────────────┐
│             **Agent Core**              │
│   – Planner ─ Researcher ─ Formatter    │
│   – Answer  ─ Coder      ─ Runner       │
│   – Feature ─ Patcher    ─ Reporter     │
│   – Decision ─ Internal Monologue       │
└─────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│               Services                  │
│  • Browser / Playwright                 │
│  • Knowledge-base (Qdrant + Redis)      │
│  • External APIs (LLMs, GitHub, etc.)   │
└─────────────────────────────────────────┘
```

* All long-running background work is executed with **asyncio** tasks coordinated
  by *OpenAI Swarm* event hand-off.
* The only persistent state is an **SQLite** database (`data/agent.db`) and a
  **Qdrant** vector store; both live on the host machine – no Docker.

---

## 2 · Configuration

* **`config.yaml`** is the single source of truth.  The helper class
  `src/config.py::Config` loads it, merges sane defaults, _then_ allows
  environment-variable overrides (great for CI or `.env`).
* All code that previously looked for `config.toml` or *tomlkit* has been
  removed.  A defensive check in `Config._load_config()` will raise an explicit
  error if the YAML file is missing so that users are never confused about the
  required format.

### Example fragment

```yaml
azure_openai:
  enabled: true
  model: gpt-4o
search_engines:
  primary: duckduckgo
```

---

## 3 · Unsandboxed Code Execution

* The **Runner Agent** delegates to `src/services/terminal_runner.TerminalRunner`.
* Instead of spawning containers (Docker) or using Firejail, the runner:
  1. Creates a **temporary working directory** (auto-cleaned).
  2. Launches the target command via `subprocess.Popen`.
  3. Applies soft **CPU & memory limits** on POSIX (via `resource`) and
     **process-group timeouts** on Windows.
* This keeps the project OS-agnostic and requires no extra system packages.

> ⚠️  The execution happens on the _host_ machine.  Use the built-in limits &
>      prompt confirmation logic in the front-end before running untrusted code.

---

## 4 · Eliminated Dependencies

| Legacy           | Replacement / Action                     |
| ---------------- | ---------------------------------------- |
| Docker / Compose | **Removed** – all scripts & docs pruned. |
| Firejail         | **Removed** – replaced by TerminalRunner |
| `tomlkit` / TOML | **YAML** via `PyYAML`                    |

---

## 5 · Persistence & Observability

* **Prometheus** metrics server is started in `main.py` using the port defined
  in `config.yaml -> monitoring.metrics.prometheus.port`.
* **OpenTelemetry** spans are exported to the console by default; wire up a
  Jaeger backend by toggling `config.yaml -> monitoring.tracing.jaeger`.
* Structured JSON logs are written to the file configured under
  `logging.log_file` (defaults to `logs/app.log`).

---

## 6 · Future work

* Allow users to choose a _per-run_ resource profile (cpu / mem / timeout).
* Add a Windows-specific job-object based memory cap once Python 3.12 lands.
* Auto-detect LLM provider pricing changes and update `config.yaml` at runtime.
