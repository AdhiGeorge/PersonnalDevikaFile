# Under-the-Hood Changes (June 2025)

This companion to `ARCHITECTURE.md` focuses on implementation-level details that
changed while removing Docker/sandboxing and the old TOML configuration.

---

## 1. Config Loader (`src/config.py`)

* The loader was already migrated to YAML; we have now **deleted** residual code
  comments referencing `config.toml`.
* Added `_apply_env_overrides()` which maps common env-vars (e.g.
  `AZURE_OPENAI_API_KEY`) onto the nested YAML structure.
* **Defaults** are declared inline and merged via `_update_nested_dict()` – no
  external sample file needed.

> Tip: call `Config().get_config()` anywhere for a full resolved dict.

---

## 2. Terminal Execution Flow

1. **UI** – user clicks "Run" → emits `user-message` with `action: run`.
2. **`main.py`** spawns a background thread to `_run_agent`.
3. **Runner Agent**
   * Formats prompt via `runner.jinja2`.
   * Calls LLM to decide which **shell commands** to execute.
   * Sends them to `TerminalRunner.run()`.
4. **TerminalRunner**
   * Creates `tempfile.TemporaryDirectory()`.
   * OS-specific resource guard:
     * **POSIX** – `resource.setrlimit()` for CPU & RSS.
     * **Windows** – `CREATE_NEW_PROCESS_GROUP` + `threading.Timer` kill.
   * Streams back JSON (stdout / stderr / code / duration).

No Docker image build, no container spin-up latency.

---

## 3. Removed Artifacts

* `setup.sh` lines referencing `docker` were purged earlier.
* `.gitignore` entry for `config.toml` is kept (harmless) to avoid resurrecting
  the file accidentally; feel free to remove if preferred.
* All README snippets are updated to mention **`config.yaml`** only.

---

## 4. Documentation Build

The docs folder is now minimal; you can generate a static site with MkDocs:

```bash
pip install mkdocs-material
mkdocs build
```

It will automatically pick up these Markdown files.
