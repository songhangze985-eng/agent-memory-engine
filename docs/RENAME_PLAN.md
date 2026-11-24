# Package Rename Migration Plan (`memengine` → `agentmem-core` / `agentmem_core`)

> **Status: Phase A EXECUTED (`v0.3.0`)** — Approved by maintainer (`agentmem-core`).
> Core distribution is now `agentmem-core` (`import agentmem_core`, CLI `agentmem-core` / `agentmem`, env prefix `AGENTMEM_CORE_*`). Legacy `import memengine`, `memengine` CLI, and `MEMENGINE_*` env vars remain 100% functional via `memengine/__init__.py` alias shim through `v0.3.x`.

## 1. Why Rename

The current Python distribution and import package name `memengine` (in `pyproject.toml`) collides directly with the published **WWW'25 MemEngine** framework (`memengine` in literature/GitHub/PyPI ecosystems). Keeping `memengine` creates:

- PyPI distribution namespace collision and installation ambiguity.
- Citation / benchmark confusion in evaluation reports and external comparisons.
- Confusing CLI binary (`memengine`) and environment variable prefixes (`MEMENGINE_*`).

> **Non-Breaking Policy**: Existing `import memengine` and `MEMENGINE_*` environment variables are **kept 100% functional** until a target name is explicitly approved by the maintainer. No breaking rename is applied unilaterally in M5.

## 2. Candidate Package Names

| Candidate | PyPI / Import Style | Positioning Fit | Pros | Cons |
| :--- | :--- | :--- | :--- | :--- |
| **`chronomem`** *(Recommended)* | `pip install chronomem` / `import chronomem` | Bi-temporal ledger + episode log + deterministic coding/conversational memory | Short, distinctive, highlights bi-temporal & chronicle identity, zero WWW'25 collision | Requires verifying PyPI availability before publishing |
| **`memledger`** | `pip install memledger` / `import memledger` | Unified evidence-backed fact ledger + derived VFS/Profile views | Clearly communicates single-truth ledger architecture vs RAG blobs | Slightly narrower than procedural rules/dreaming |
| **`agentmem-core`** | `pip install agentmem-core` / `import agentmem_core` | General-purpose agent memory kernel | Self-explanatory | Underscore vs hyphen friction; generic namespace |

## 3. Zero-Downtime Two-Phase Migration Strategy

### Phase A — Dual-Export Release (`v0.3.0`, Zero Breaking Changes)

1. Rename core package directory `memengine/` → `<new_pkg>/`.
2. Update `pyproject.toml`:
   - `project.name = "<new_pkg>"`
   - Register both CLI entrypoints: `<new_pkg> = "<new_pkg>.cli:main"` and deprecated alias `memengine = "<new_pkg>.cli:main"`.
   - Include both packages in `[tool.setuptools.packages.find]`: `include = ["<new_pkg>*", "memengine*"]`.
3. Provide a thin compatibility shim package `memengine/__init__.py` (and lazy submodule loader via PEP 562 `__getattr__` / `sys.meta_path` alias):
   - Emits `DeprecationWarning("The 'memengine' package is renamed to '<new_pkg>' and will be removed in v1.0.")`.
   - Forwards every attribute and submodule import (`from memengine.core import UnifiedMemoryEngine`, `from memengine import CognitiveMemoryEngine`, etc.) to `<new_pkg>.*`.
4. Environment variables (`<NEW_PKG>_DB_URL`, `<NEW_PKG>_API_KEY`, etc.):
   - Read `<NEW_PKG>_*` first, fall back to `MEMENGINE_*` with a one-time deprecation warning.
5. HTTP API routes:
   - Unified routes `/v2/memory/*` remain unchanged; deprecated `/v1/*` shim (`Deprecation: true` header) stays through `v0.3.x`.

### Phase B — Removal Release (`v1.0.0`)

1. Remove the `memengine/` compatibility shim package and `MEMENGINE_*` env fallbacks.
2. Remove the deprecated `/v1/*` HTTP shim (`compat/v1_shim.py`) and legacy `VFSMemoryEngine` parallel storage path (`vfs/`).
3. Publish `<new_pkg>==1.0.0` with clean single-core API surface.

## 4. Exact Scope Checklist (Repo Inventory)

When a target name is approved, execute the rename across:

- `pyproject.toml` (`name`, `[project.scripts]`, `[tool.setuptools.packages.find]`, `data-files`)
- `memengine/` → `<new_pkg>/` + compatibility shim `memengine/`
- `tests/` & `eval/` internal imports
- `scripts/` & `docker-compose.yml` / `.github/workflows/ci.yml`
- `README.md`, `docs/DESIGN.md`, `docs/SPEC.md`
