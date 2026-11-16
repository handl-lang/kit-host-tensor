"""``kit_host_tensor.wire`` — the f64 wire codec, extension-free.

Mirrors host-tensor ``src/ops/codec.rs`` (commit ``0528649``):
non-finite ``f64`` elements cross as ``{"__nonfinite__": "nan" |
"inf" | "-inf"}`` tag objects inside ``data``; finite tensors encode
byte-identically to the pre-tag wire form.  These tests need no PyO3
extension — they exercise the Python codec directly.
"""

from __future__ import annotations

import base64
import warnings

import numpy as np
import pytest
from kit_host_tensor.errors import HostTensorError
from kit_host_tensor.wire import (
    NONFINITE_TAG,
    TENSOR_BIN_TAG,
    WIRE_TAG,
    from_wire,
    to_wire,
)


def test_finite_tensor_encodes_byte_identical_wire_form() -> None:
    """Finite payloads keep the pre-tag spelling: plain numbers."""
    rec = to_wire(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert rec == {
        WIRE_TAG: {
            "shape": [2, 2],
            "data": [1.0, 2.0, 3.0, 4.0],
            "dtype": "f64",
        }
    }
    assert from_wire(rec).shape == (2, 2)
    np.testing.assert_allclose(from_wire(rec), [[1, 2], [3, 4]])


def test_nonfinite_elements_emit_and_decode_tags() -> None:
    """NaN/±inf encode as ``__nonfinite__`` tags and decode back."""
    rec = to_wire(np.array([1.0, np.nan, np.inf, -np.inf]))
    data = rec[WIRE_TAG]["data"]
    assert data[0] == 1.0
    assert data[1:] == [
        {NONFINITE_TAG: "nan"},
        {NONFINITE_TAG: "inf"},
        {NONFINITE_TAG: "-inf"},
    ]
    out = from_wire(rec)
    assert out[0] == 1.0
    assert np.isnan(out[1])
    assert out[2] == np.inf and out[3] == -np.inf


def test_from_wire_accepts_rust_spelled_tags() -> None:
    """The Rust ``decode_f64`` spellings decode identically."""
    rec = {
        WIRE_TAG: {
            "shape": [3],
            "data": [
                {"__nonfinite__": "nan"},
                {"__nonfinite__": "inf"},
                {"__nonfinite__": "-inf"},
            ],
            "dtype": "f64",
        }
    }
    out = from_wire(rec)
    assert np.isnan(out[0]) and out[1] == np.inf and out[2] == -np.inf


def test_from_wire_rejects_malformed_elements_loudly() -> None:
    """``null`` (the pre-tag spelling), strings, bools, and unknown
    tags are loud errors — never silent coercions."""
    for bad in (
        None,
        "1.5",
        True,
        {"__nonfinite__": "bogus"},
        {"x": 1},
    ):
        rec = {WIRE_TAG: {"shape": [1], "data": [bad], "dtype": "f64"}}
        with pytest.raises(HostTensorError, match="data element"):
            from_wire(rec)


def test_to_wire_warns_on_non_f64_ndarray() -> None:
    """The f64 upcast is honest: non-float64 ndarrays warn, the
    ``"f64"`` stamp describes the payload, not the source."""
    with pytest.warns(UserWarning, match="upcasts float32"):
        rec = to_wire(np.array([1.0, 2.0], dtype=np.float32))
    assert rec[WIRE_TAG]["dtype"] == "f64"
    # float64 input does not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        to_wire(np.array([1.0, 2.0], dtype=np.float64))


# ---------------------------------------------------------------------------
#  ADR-0179 binary ``data`` spelling — ``{"__tensor_bin__": "<b64 LE>"}``
# ---------------------------------------------------------------------------


def _bin_rec(shape: list[int], dtype: str, payload: bytes) -> dict:
    return {
        WIRE_TAG: {
            "shape": shape,
            "data": {
                TENSOR_BIN_TAG: base64.b64encode(payload).decode("ascii")
            },
            "dtype": dtype,
        }
    }


def test_from_wire_decodes_f64_bin_payload() -> None:
    """The Rust codecs emit ``__tensor_bin__`` for numel >= 4096 —
    decode must read it back exactly."""
    arr = np.arange(12.0, dtype="<f8").reshape(3, 4)
    rec = _bin_rec([3, 4], "f64", arr.tobytes())
    out = from_wire(rec)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, arr)


def test_from_wire_decodes_f32_bin_payload_widened() -> None:
    """``"dtype": "f32"`` payloads widen ``f32 → f64`` exactly —
    every f32 is an f64."""
    arr32 = np.array([0.5, -1.25, 3.0], dtype="<f4")
    rec = _bin_rec([3], "f32", arr32.tobytes())
    out = from_wire(rec)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, arr32.astype(np.float64))


def test_from_wire_bin_rejects_bad_dtype() -> None:
    """Only ``"f32"``/``"f64"`` are accepted — anything else is loud."""
    rec = _bin_rec([1], "f16", b"\x00" * 2)
    with pytest.raises(HostTensorError, match="dtype"):
        from_wire(rec)


def test_from_wire_bin_rejects_bad_base64() -> None:
    """Malformed base64 is a loud error, never a silent decode."""
    rec = {
        WIRE_TAG: {
            "shape": [1],
            "data": {TENSOR_BIN_TAG: "!!!not-base64!!!"},
            "dtype": "f64",
        }
    }
    with pytest.raises(HostTensorError, match="base64"):
        from_wire(rec)


def test_from_wire_bin_rejects_wrong_payload_length() -> None:
    """Payload must be ``numel x itemsize`` bytes — a truncated or
    padded payload is a loud error."""
    rec = _bin_rec([2, 2], "f64", b"\x00" * 16)  # needs 32
    with pytest.raises(HostTensorError, match="bytes"):
        from_wire(rec)


def test_from_wire_bin_requires_bin_tag_value() -> None:
    """A dict ``data`` without a base64 string under ``__tensor_bin__``
    is rejected — the array-spelling decoder must not see dict keys."""
    rec = {
        WIRE_TAG: {
            "shape": [1],
            "data": {"other": "x"},
            "dtype": "f64",
        }
    }
    with pytest.raises(HostTensorError):
        from_wire(rec)
