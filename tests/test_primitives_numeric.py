"""Real numeric correctness tests for the Tensor-backed primitive layer.

Every primitive exercised here (Tensor, add/mul/div, exp/rsqrt/silu,
sum/max, matmul, reshape/view, transpose/permute, concat/stack,
repeat/expand, slice/gather, where, cast) is pure real numpy -- none of
these tests launch Docker/TimeLoop (see wafersim/primitives/tensor's
module docstring for why that's a real, structural fact, not a shortcut
taken for testing convenience). This mirrors wafer/functional.py's own
existing convention of checking a real schedule against a plain numpy
reference (e.g. ``A @ B``) rather than a synthetic/assumed answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from wafersim.primitives.add import add
from wafersim.primitives.cast import cast
from wafersim.primitives.concat_stack import concat, stack
from wafersim.primitives.div import div
from wafersim.primitives.exp import exp
from wafersim.primitives.matmul import matmul
from wafersim.primitives.max import max as t_max
from wafersim.primitives.mul import mul
from wafersim.primitives.repeat_expand import expand, repeat
from wafersim.primitives.reshape_view import reshape, view
from wafersim.primitives.rsqrt import rsqrt
from wafersim.primitives.silu import silu
from wafersim.primitives.slice_gather import gather, slice_
from wafersim.primitives.sum import sum as t_sum
from wafersim.primitives.tensor import Tensor
from wafersim.primitives.transpose_permute import permute, transpose
from wafersim.primitives.where_select import where


# ---- Tensor itself --------------------------------------------------

def test_tensor_construction_and_introspection():
    t = Tensor.from_values([[1.0, 2.0], [3.0, 4.0]])
    assert t.shape == (2, 2)
    assert t.dtype == np.float32
    assert t.bytes_per_element == 4
    assert t.size == 4


def test_tensor_rejects_non_array():
    with pytest.raises(TypeError):
        Tensor([1.0, 2.0])  # raw list, not an ndarray


def test_tensor_rejects_non_numeric_dtype():
    with pytest.raises(TypeError):
        Tensor(np.array(["a", "b"]))


def test_tensor_allclose():
    a = Tensor.from_values([1.0, 2.0, 3.0])
    b = Tensor.from_values([1.0, 2.0, 3.0])
    c = Tensor.from_values([1.0, 2.0, 3.1])
    assert a.allclose(b)
    assert not a.allclose(c)


# ---- arithmetic -------------------------------------------------------

def test_add_matches_numpy_reference():
    a = Tensor.random((4, 4), seed=1)
    b = Tensor.random((4, 4), seed=2)
    result = add(a, b)
    assert np.allclose(result.data, a.data + b.data)


def test_mul_matches_numpy_reference():
    a = Tensor.random((3, 5), seed=3)
    b = Tensor.random((3, 5), seed=4)
    result = mul(a, b)
    assert np.allclose(result.data, a.data * b.data)


def test_div_matches_numpy_reference():
    a = Tensor.random((3, 5), seed=5)
    b = Tensor.from_values(np.abs(np.random.default_rng(6).standard_normal((3, 5))) + 0.5)
    result = div(a, b)
    assert np.allclose(result.data, a.data / b.data)


def test_arithmetic_rejects_dtype_mismatch():
    a = Tensor.zeros((2,), dtype=np.float32)
    b = Tensor.zeros((2,), dtype=np.float16)
    with pytest.raises(TypeError):
        add(a, b)
    with pytest.raises(TypeError):
        mul(a, b)
    with pytest.raises(TypeError):
        div(a, b)


# ---- unary --------------------------------------------------------------

def test_exp_matches_numpy_reference():
    x = Tensor.random((4,), seed=7)
    assert np.allclose(exp(x).data, np.exp(x.data))


def test_rsqrt_matches_numpy_reference():
    x = Tensor.from_values([1.0, 4.0, 16.0, 100.0])
    assert np.allclose(rsqrt(x).data, 1.0 / np.sqrt(x.data))


def test_silu_matches_numpy_reference():
    x = Tensor.random((5,), seed=8)
    expected = x.data / (1.0 + np.exp(-x.data))
    assert np.allclose(silu(x).data, expected)


# ---- reduction ------------------------------------------------------

def test_sum_matches_numpy_reference_and_keeps_dims():
    x = Tensor.random((2, 6), seed=9)
    result = t_sum(x, dim=-1)
    assert result.shape == (2, 1)
    assert np.allclose(result.data, np.sum(x.data, axis=-1, keepdims=True))


def test_max_matches_numpy_reference_and_keeps_dims():
    x = Tensor.random((2, 6), seed=10)
    result = t_max(x, dim=-1)
    assert result.shape == (2, 1)
    assert np.allclose(result.data, np.max(x.data, axis=-1, keepdims=True))


# ---- linear algebra ---------------------------------------------------

def test_matmul_matches_numpy_reference():
    a = Tensor.random((4, 3), seed=11)
    b = Tensor.random((3, 5), seed=12)
    result = matmul(a, b)
    assert np.allclose(result.data, a.data @ b.data)


# ---- shape / memory ---------------------------------------------------

def test_reshape_and_view_real_distinction():
    x = Tensor.from_values(np.arange(12, dtype=np.float32))
    r = reshape(x, (3, 4))
    v = view(x, (3, 4))
    assert r.shape == (3, 4) and v.shape == (3, 4)
    assert np.allclose(r.data, v.data)

    # A transposed tensor is not viewable into a different shape without
    # copying -- view() must raise, matching real numpy memory-layout
    # semantics (not a fabricated restriction).
    xt = transpose(reshape(x, (3, 4)), 0, 1)  # shape (4, 3), non-contiguous
    with pytest.raises(ValueError):
        view(xt, (12,))
    # reshape() is allowed to copy, so it still succeeds.
    assert reshape(xt, (12,)).shape == (12,)


def test_permute_and_transpose_match_numpy():
    x = Tensor.random((2, 3, 4), seed=13)
    p = permute(x, (2, 0, 1))
    assert p.shape == (4, 2, 3)
    assert np.allclose(p.data, np.transpose(x.data, axes=(2, 0, 1)))

    t = transpose(x, 0, 2)
    assert t.shape == (4, 3, 2)
    assert np.allclose(t.data, np.transpose(x.data, axes=(2, 1, 0)))


def test_concat_and_stack_match_numpy():
    a = Tensor.random((2, 3), seed=14)
    b = Tensor.random((2, 3), seed=15)
    c = concat([a, b], dim=0)
    assert c.shape == (4, 3)
    assert np.allclose(c.data, np.concatenate([a.data, b.data], axis=0))

    s = stack([a, b], dim=0)
    assert s.shape == (2, 2, 3)
    assert np.allclose(s.data, np.stack([a.data, b.data], axis=0))


def test_concat_rejects_dtype_mismatch():
    a = Tensor.zeros((2, 2), dtype=np.float32)
    b = Tensor.zeros((2, 2), dtype=np.float16)
    with pytest.raises(TypeError):
        concat([a, b], dim=0)


def test_repeat_and_expand_match_numpy_and_broadcast_semantics():
    x = Tensor.random((2, 1, 4), seed=16)
    r = repeat(x, 3, dim=1)
    assert r.shape == (2, 3, 4)
    assert np.allclose(r.data, np.repeat(x.data, 3, axis=1))

    e = expand(x, (2, 3, -1))
    assert e.shape == (2, 3, 4)
    # expand is a real broadcast view: every "repeated" row is identical to
    # the original single row (GQA K/V-head broadcast semantics).
    assert np.allclose(e.data[:, 0, :], e.data[:, 1, :])
    assert np.allclose(e.data[:, 0, :], x.data[:, 0, :])


def test_slice_and_gather_match_numpy():
    x = Tensor.from_values(np.arange(20, dtype=np.float32).reshape(4, 5))
    s = slice_(x, dim=1, start=1, end=3)
    assert s.shape == (4, 2)
    assert np.allclose(s.data, x.data[:, 1:3])

    g = gather(x, dim=0, index=[0, 2, 2])
    assert g.shape == (3, 5)
    assert np.allclose(g.data, x.data[[0, 2, 2], :])


# ---- mask / type --------------------------------------------------------

def test_where_matches_numpy_reference_causal_mask():
    scores = Tensor.random((4, 4), seed=17)
    # 0/1 float causal mask, matching this project's own documented
    # convention (see wafersim/primitives/tensor's module docstring).
    mask = Tensor.from_values(np.tril(np.ones((4, 4), dtype=np.float32)))
    neg_inf = Tensor.from_values(np.full((4, 4), -1e9, dtype=np.float32))
    masked = where(mask, scores, neg_inf)
    expected = np.where(mask.data != 0, scores.data, neg_inf.data)
    assert np.allclose(masked.data, expected)


def test_cast_changes_dtype_and_values():
    x = Tensor.from_values([1.0, 2.0, 3.0], dtype=np.float32)
    y = cast(x, np.float16)
    assert y.dtype == np.float16
    assert np.allclose(y.data.astype(np.float32), x.data)


# ---- end-to-end: real softmax and RMSNorm built purely from these
# primitives, checked against independent numpy reference implementations.
# This is the actual point of building Tensor + these primitives -- being
# able to compose a real forward-pass building block and get the right
# numbers out. ------------------------------------------------------------

def _softmax_via_primitives(x: Tensor) -> Tensor:
    row_max = t_max(x, dim=-1)  # (..., 1), numerical-stability shift
    shifted = add(x, Tensor.from_values(-row_max.data))
    z = exp(shifted)
    denom = t_sum(z, dim=-1)  # (..., 1)
    return div(z, denom)


def test_softmax_composed_from_primitives_matches_reference():
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((3, 6)).astype(np.float32)
    x = Tensor.from_values(raw)

    result = _softmax_via_primitives(x)

    # independent numpy reference softmax
    shifted = raw - raw.max(axis=-1, keepdims=True)
    expected = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)

    assert result.shape == x.shape
    assert np.allclose(result.data, expected, atol=1e-5, rtol=1e-5)
    # rows must each sum to 1
    assert np.allclose(result.data.sum(axis=-1), 1.0, atol=1e-5)


def _rmsnorm_via_primitives(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    n = x.shape[-1]
    sum_sq = t_sum(mul(x, x), dim=-1)  # (..., 1)
    mean_sq = div(sum_sq, Tensor.from_values(np.full(sum_sq.shape, float(n), dtype=x.dtype)))
    eps_t = Tensor.from_values(np.full(mean_sq.shape, eps, dtype=x.dtype))
    normed = mul(x, rsqrt(add(mean_sq, eps_t)))
    return mul(normed, weight)


def test_rmsnorm_composed_from_primitives_matches_reference():
    rng = np.random.default_rng(123)
    raw = rng.standard_normal((2, 8)).astype(np.float32)
    w = np.ones((8,), dtype=np.float32) * 1.5
    x = Tensor.from_values(raw)
    weight = Tensor.from_values(w)

    result = _rmsnorm_via_primitives(x, weight)

    eps = 1e-6
    mean_sq = (raw * raw).mean(axis=-1, keepdims=True)
    expected = raw / np.sqrt(mean_sq + eps) * w

    assert result.shape == x.shape
    assert np.allclose(result.data, expected, atol=1e-4, rtol=1e-4)
