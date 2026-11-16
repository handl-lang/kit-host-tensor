# Agent guide — kit-host-tensor

The host-tensor backend for the mlopt stack — `HostTensorSink`, an
`mlopt_core.ports.Sink` that lowers the 138 liftable-now IR ops onto
the `handl-lang/host-tensor` Rust `tensor.*` op registry through its
PyO3 binding (`handl_host_tensor.call`), plus `HostTensorExecutor`,
the bottom-up term interpreter, the f64 wire codec (`wire.py` —
`__tensor__` records, `__nonfinite__` element tags, ADR-0179
`__tensor_bin__` binary `data`), and the generated `host_surface.json`
op-surface artifact.  Split from `handl-lang/graphopt`
`packages/mlopt-host-tensor` in P3 Step D-1f (the `mlopt-*` →
`kit-*` rename, history preserved via `git subtree split`).
Python 3.13, `mlopt-core` on the eqsat substrate + numpy.

## Layout

- `src/kit_host_tensor/` — the one shipped package:
  - `sink.py` — `HostTensorSink` (`mlopt_core.ports.Sink` impl:
    `lower`/`verify`/`supported_ops`);
  - `executor.py` — `HostTensorExecutor`, the bottom-up interpreter
    over `handl_host_tensor.call`;
  - `bindings.py` — `HOST_BINDINGS`, the 138-op IR→host
    positional-arg table (validated against `host_surface.json`);
  - `wire.py` — the `{"__tensor__": …}` f64 wire codec +
    `__nonfinite__`/`__tensor_bin__` (ADR-0179) decode;
  - `surface.py` — `host_surface.json` reader + the host-tensor
    checkout surface collector (regen half of
    `scripts/dump_host_surface.py`);
  - `_binding.py` — the `handl_host_tensor` PyO3 ext loader
    (env → import → sibling `.so`);
  - `report.py` — `HostTensorVerifyReport` + NumPy equivalence gate;
  - `errors.py` — `HostTensorError` / `UnsupportedHostOpError`.
- `scripts/dump_host_surface.py` — regenerates the package's own
  `host_surface.json` artifact from a sibling host-tensor checkout
  (`HOST_TENSOR_DIR` or `../host-tensor`; `--check` is the freshness
  gate).
- `tests/` — the package-bound suite carved out of graphopt's
  `tests/`: `test_host_tensor_sink.py` (extension-gated — skips
  cleanly without the `.so`), `test_host_tensor_wire.py`
  (extension-free codec incl. `__tensor_bin__`),
  `test_host_surface.py` (artifact shape + bindings drift +
  freshness check when the checkout is present).  Cross-stack tests
  (`test_surface_table_drift.py`, `test_onnx_codec_drift.py`,
  `test_hnl_import.py`) stayed in graphopt and import this package
  as a git dep.

## Contracts

- **Leaf backend** — deps are `mlopt-core` (handl-lang/eqsat,
  `packages/mlopt-core` subdir, tag-pinned in
  `[tool.uv.sources]`) + `numpy>=1.24`.  Nothing else: no torch, no
  `mlopt_orchestrator`, no gamekit, no sibling `kit-*`.  The
  import-linter `forbidden` contract enforces it (both `mlopt_*` and
  `kit_*` spellings are listed during the transition).
- **`handl_host_tensor` is an undeclared runtime, not a pip dep** —
  `_binding.py` resolves the PyO3 extension in order:
  `$KIT_HOST_TENSOR_LIB` → `import handl_host_tensor` →
  `<repo>/../host-tensor/target/{debug,release}/handl_host_tensor.so`
  (`$KIT_HOST_TENSOR_REPO` overrides the checkout path,
  `$KIT_HOST_TENSOR_TARGET` the cargo profile).  The env vars were
  renamed `MLOPT_HOST_TENSOR_*` → `KIT_HOST_TENSOR_*` with the
  package — update dev environments accordingly.  `HOST_TENSOR_DIR`
  (unchanged) points the surface collector at the host-tensor
  checkout.  Import of the package never requires the Rust build;
  only *using* the sink does, and every failure is a loud
  `HostTensorError` with the build command.
- **Sibling layout** — the repo assumes `kit-host-tensor` and
  `handl-lang/host-tensor` are siblings under `handl-lang/`;
  `_binding.py` `_REPO_ROOT` is `parents[2]` of
  `src/kit_host_tensor/_binding.py`, `surface.py`'s default
  `HOST_TENSOR_DIR` is `parents[3]/host-tensor`.  Both keep working
  in any checkout nested one level under a dir containing
  `host-tensor` — env vars override regardless.
- **Honest codec** — f64-only wire; non-finite elements cross as
  `{"__nonfinite__": …}` tags; `__tensor_bin__` decodes but
  `to_wire` always emits the array spelling; malformed records are
  loud `HostTensorError`s, never silent coercions.
- **`host_surface.json` is generated** — do not hand-edit;
  `scripts/dump_host_surface.py` (or `--check`) against the
  host-tensor checkout.  `test_host_surface.py` enforces freshness
  when the checkout is present.

## Verification

```sh
uv sync --group dev
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src
.venv/bin/ruff format --check src
.venv/bin/ty check
.venv/bin/lint-imports
coverage run -m pytest tests/ -q && coverage report -m   # floor in pyproject
```

<!-- myco:agent-awareness -->

## Agent awareness

- Session start / any work discussion: run `myco status --prompt`
  before proposing next steps — know the board first.
- User describes new work → offer to capture it (`myco todo add`,
  `issue new`, `proposal new`). Don't wait to be asked.
- Picking up a task: `myco resume <ID>` is the briefing.
- Wrapping up: `myco handoff <ID>` anything resumable; keep statuses
  honest (claimed → in-progress → blocked/done).

<!-- /myco:agent-awareness -->
