# kit-host-tensor

The host-tensor backend for mlopt — `mlopt_core.ports.Sink` over the
`handl-lang/host-tensor` Rust `tensor.*` op surface via its PyO3
binding — split out of `handl-lang/graphopt`
(`packages/mlopt-host-tensor`) in P3 Step D-1f and renamed
`mlopt-host-tensor` → `kit-host-tensor` /
`mlopt_host_tensor` → `kit_host_tensor`.

`kit_host_tensor.HostTensorSink` lowers the 138 liftable-now IR ops
(elementwise / reduce-dim / linalg / shape / constructors) onto
positional `tensor.*` calls, and `HostTensorExecutor` interprets the
lowered term bottom-up through `handl_host_tensor.call`.  The codec
is the f64 row-major wire record
`{"__tensor__": {"shape": [...], "data": [...], "dtype": "f64"}}` on
both sides, with `{"__nonfinite__": …}` element tags (NaN/±inf) and
the ADR-0179 `{"__tensor_bin__": "<base64 LE>"}` binary `data`
spelling for large payloads.

It is a **leaf backend**: it depends on `mlopt-core` (handl-lang/eqsat)
+ `numpy` only — never on torch, the orchestrator, the game layer, or
any sibling `kit-*` backend.  The `handl_host_tensor` PyO3 extension
is an *undeclared runtime*: resolved by
`$KIT_HOST_TENSOR_LIB` → `import handl_host_tensor` → the sibling
`../host-tensor/target/<profile>/` checkout (env overrides
`$KIT_HOST_TENSOR_REPO` / `$KIT_HOST_TENSOR_TARGET`,
`$HOST_TENSOR_DIR` for the surface collector).  Importing the package
never requires the Rust build — only *using* the sink does; every
failure raises `HostTensorError` with the build command.

`host_surface.json` is the generated `tensor.*` op-surface artifact
(239 ops: `host-tensor/src/index.hnl` handwritten decls +
`aten_port_gen --emit-spec`); regenerate with
`scripts/dump_host_surface.py` against a sibling host-tensor
checkout.

## Install

```sh
uv sync --group dev
```

or as a git dependency:

```toml
kit-host-tensor = { git = "https://github.com/handl-lang/kit-host-tensor.git", tag = "v0.1.0" }
```

Substrate pins (`mlopt-core`) resolve from `handl-lang/eqsat` via
this repo's `[tool.uv.sources]`; consumers' own sources win when they
git-pin the package.

## Verify

```sh
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check src
.venv/bin/ty check
.venv/bin/lint-imports
coverage run -m pytest tests/ -q && coverage report -m   # floor in pyproject
```

Tests that touch the PyO3 extension skip cleanly when the `.so`
isn't built; the wire codec, bindings table, and surface artifact
tests run extension-free.
