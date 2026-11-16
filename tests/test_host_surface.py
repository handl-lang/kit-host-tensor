"""The host ``tensor.*`` surface artifact — drift guards.

``kit_host_tensor/host_surface.json`` is the graphopt-side copy of
the authoritative host-tensor op surface (``host-tensor/src/index.hnl``
handwritten decls + ``aten_port_gen --emit-spec``).  These tests pin:

* the artifact loads and covers the ops the bridge tables name;
* every ``HOST_BINDINGS`` key is a real ``tensor.*`` op — the IR→host
  table cannot name an op the host does not carry;
* when the host-tensor checkout is present, the artifact is *fresh* —
  ``collect_host_surface`` regenerates byte-identical JSON (the same
  discipline ``aten_port_gen --check`` enforces inside host-tensor).
"""

from __future__ import annotations

import pytest

from kit_host_tensor.surface import (
    collect_host_surface,
    host_ops,
    render_surface,
    surface_json_path,
)


def _host_tensor_available() -> bool:
    try:
        collect_host_surface()
    except (FileNotFoundError, RuntimeError):
        return False
    return True


class TestArtifact:
    def test_surface_loads_and_is_shaped(self) -> None:
        ops = host_ops()
        assert len(ops) > 200
        for name, entry in ops.items():
            assert name.startswith("tensor.")
            assert entry["op"] == name
            assert entry["signature"]
            assert entry["kind"] in {"handwritten", "generated"}

    def test_required_ops_present(self) -> None:
        ops = host_ops()
        for name in (
            "tensor.add",
            "tensor.matmul",
            "tensor.conv2d",
            "tensor.batch_norm",
            "tensor.max_pool",
            "tensor.reshape",
            "tensor.transpose",
            "tensor.concat",
            "tensor.softmax",
            "tensor.layer_norm",
            "tensor.from_list",
            "tensor.shape",
        ):
            assert name in ops, name

    def test_tensor_shape_composite_result(self) -> None:
        """``tensor.shape : Tensor -> (List Int)`` — the multi-token
        result type that a single-``\\S+`` decl regex silently drops."""
        entry = host_ops()["tensor.shape"]
        assert entry["signature"] == "Tensor -> (List Int)"
        assert entry["kind"] == "handwritten"

    def test_op_count(self) -> None:
        """239 ops — the full host-tensor ``index.hnl`` decl surface."""
        assert len(host_ops()) == 239

    def test_generated_and_handwritten_split(self) -> None:
        kinds = {e["kind"] for e in host_ops().values()}
        assert kinds == {"handwritten", "generated"}


class TestBindingsDrift:
    """``HOST_BINDINGS`` keys must be real ``tensor.*`` ops."""

    def test_every_binding_key_is_a_host_op(self) -> None:
        from kit_host_tensor.bindings import HOST_BINDINGS

        surface = host_ops()
        missing = {f"tensor.{k}" for k in HOST_BINDINGS} - set(surface)
        assert not missing, (
            f"HOST_BINDINGS names ops the host surface lacks: "
            f"{sorted(missing)}"
        )

    def test_binding_count_stable(self) -> None:
        from kit_host_tensor.bindings import HOST_BINDINGS

        assert len(HOST_BINDINGS) == 138

    # NOTE: graphopt's ``test_host_surface.py`` also carried
    # ``test_import_covered_ops_are_host_ops`` — the
    # ``kit_discovery.hnl_import.covered_ops`` ↔ host-surface drift
    # check.  Re-homing decision (D-1f Phase 2): it stays in
    # graphopt's suite — no kit repo's dep closure contains both
    # ``kit_discovery.hnl_import`` and ``kit_host_tensor.surface``
    # (each forbids the other in its import-linter contract), and
    # graphopt's post-Phase-3 closure has both.


@pytest.fixture(scope="module")
def _fresh_surface_text() -> str | None:
    if not _host_tensor_available():
        return None
    return render_surface(collect_host_surface())


class TestFreshness:
    """The artifact must match a fresh collect when the checkout is
    available — the generate-don't-handwrite drift gate."""

    def test_artifact_is_fresh(
        self, _fresh_surface_text: str | None
    ) -> None:
        if _fresh_surface_text is None:
            pytest.skip(
                "host-tensor checkout/binary unavailable — "
                "set HOST_TENSOR_DIR"
            )
        current = surface_json_path().read_text()
        assert current == _fresh_surface_text, (
            "host_surface.json is stale — run "
            ".venv/bin/python scripts/dump_host_surface.py"
        )
