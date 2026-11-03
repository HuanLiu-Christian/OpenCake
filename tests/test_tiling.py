"""Unit tests for wafersim.tiling (pure Python, no Docker)."""

from __future__ import annotations

import pytest

from wafersim.tiling import Shard, Tiling


def test_serial_is_one_pe_and_energy_multiplier_one():
    s = Tiling.serial().elements(126_720, 14_400)
    assert (s.total, s.per_pe, s.pes) == (126_720, 126_720, 1)
    assert s.energy_multiplier == 1.0


def test_mesh_spreads_over_pes_and_conserves_total_work():
    s = Tiling().elements(126_720, 14_400)
    assert s.pes <= 14_400 and s.per_pe == 9  # ceil(126720/14400)
    assert s.per_pe * s.pes >= s.total
    assert s.energy_multiplier * s.per_pe == pytest.approx(s.total)  # no ceil inflation


def test_mesh_small_tensor_uses_at_most_n_pes():
    s = Tiling().elements(5, 14_400)
    assert (s.per_pe, s.pes) == (1, 5)


def test_min_elements_per_pe_is_a_floor_on_every_pe():
    s = Tiling(min_elements_per_pe=64).elements(1_000, 14_400)
    assert (s.per_pe, s.pes) == (67, 15)  # floor(1000/64)=15 PEs -> ceil(1000/15)=67 each
    assert s.per_pe >= 64
    # fewer elements than the minimum: stays on one PE
    assert Tiling(min_elements_per_pe=64).elements(10, 14_400).pes == 1
    # the cap is still total_pes
    assert Tiling(min_elements_per_pe=2).elements(10**6, 100).pes == 100


def test_ceil_division_busiest_pe_is_per_pe():
    s = Tiling().elements(10, 4)
    assert (s.per_pe, s.pes) == (3, 4)
    s2 = Tiling().elements(9, 4)
    assert (s2.per_pe, s2.pes) == (3, 3)  # a 4th PE would add nothing


def test_lanes_are_never_split_and_minimum_is_one_per_pe():
    s = Tiling(min_elements_per_pe=64).lanes(10, 14_400)  # min_elements ignored for lanes
    assert (s.per_pe, s.pes) == (1, 10)
    s2 = Tiling().lanes(31_702, 14_400)
    assert s2.per_pe == 3 and s2.pes <= 14_400
    assert Tiling.serial().lanes(31_702, 14_400).per_pe == 31_702


def test_more_pes_never_increases_latency():
    prev = None
    for pes in (1, 2, 8, 64, 1024, 14_400):
        per = Tiling().elements(100_000, pes).per_pe
        assert prev is None or per <= prev
        prev = per


def test_validation():
    with pytest.raises(ValueError):
        Tiling(mode="bogus")
    with pytest.raises(ValueError):
        Tiling(min_elements_per_pe=0)
    with pytest.raises(ValueError):
        Tiling().elements(0, 10)
    with pytest.raises(ValueError):
        Tiling().lanes(5, 0)
    assert isinstance(Tiling().elements(3, 3), Shard)


def test_describe_names_the_bracket():
    assert "pessimistic" in Tiling.serial().describe(100)
    assert "optimistic" in Tiling().describe(100)
