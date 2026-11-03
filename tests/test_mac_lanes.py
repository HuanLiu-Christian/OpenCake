"""mac_lanes / sram_read_bytes_per_cycle: ideal-SIMD what-if applied in wafer/simulator.py (not TimeLoop-measured)."""
import dataclasses

from wafersim.contract import GemmMappingContract
from wafersim.specs import GemmSpec, WaferSpec
from wafersim.wafer.simulator import simulate_gemm


def _contract(wafer, m=64, n=64, k=128, rows=2, cols=2):
    tm, tn = m // rows, n // cols
    return GemmMappingContract(
        gemm=GemmSpec(m=m, n=n, k=k, bytes_per_element=2), wafer=wafer, active_rows=rows, active_cols=cols, tile_m=tm, tile_n=tn,
        tile_k=64, mapper_mode="test", map_path="t.map.yaml", stats_path="t.stats.txt",
        local_compute_cycles=tm * tn * k, local_energy_uj=0.0, mac_area_um2=1.0, pe_scratchpad_area_um2=1.0,
        fifo_area_um2=1.0, router_area_um2=1.0)


BASE = WaferSpec(mesh_rows=2, mesh_cols=2, sram_kib_per_pe=48)


def test_defaults_are_the_historical_model():
    assert BASE.mac_lanes == 1 and BASE.sram_read_bytes_per_cycle == 0
    r = simulate_gemm(_contract(BASE))
    assert r.compute_busy_cycles == r.mac_operations  # 1.0 cycle/MAC


def test_eight_lanes_divide_compute_by_eight():
    w8 = dataclasses.replace(BASE, mac_lanes=8)
    r1, r8 = simulate_gemm(_contract(BASE)), simulate_gemm(_contract(w8))
    assert r8.mac_operations == r1.mac_operations
    assert r8.compute_busy_cycles * 8 == r1.compute_busy_cycles
    assert r8.sim_time_cycles < r1.sim_time_cycles  # network does not shrink, so total time is not 8x smaller
    assert r8.sim_time_cycles > r1.sim_time_cycles / 8


def test_lanes_round_up_per_phase():
    w = dataclasses.replace(BASE, mac_lanes=8)
    r = simulate_gemm(_contract(w, m=2, n=2, k=3, rows=1, cols=1))  # 12 MACs, 1 phase of 3 -> ceil per phase
    assert r.compute_busy_cycles >= -(-r.mac_operations // 8)


def test_sram_read_floor_binds_for_gemv_only():
    # m=1 per PE: each weight is read once per MAC; 8 lanes need 16 B/cycle of B. With 8 B/cycle the floor is 2x compute.
    w = dataclasses.replace(BASE, mac_lanes=8, sram_read_bytes_per_cycle=8)
    wide = dataclasses.replace(BASE, mac_lanes=8, sram_read_bytes_per_cycle=16)
    c = lambda wf: _contract(wf, m=2, n=64, k=128, rows=2, cols=2)  # tile_m = 1
    base8 = simulate_gemm(c(dataclasses.replace(BASE, mac_lanes=8))).compute_busy_cycles
    assert simulate_gemm(c(wide)).compute_busy_cycles == base8     # 16 B/cycle exactly feeds 8 lanes at tile_m=1
    assert simulate_gemm(c(w)).compute_busy_cycles == 2 * base8    # 8 B/cycle halves it
    # tile_m = 32: weights are reused 32x, so even 8 B/cycle does not bind
    big = lambda wf: _contract(wf, m=64, n=64, k=128)
    assert (simulate_gemm(big(w)).compute_busy_cycles
            == simulate_gemm(big(dataclasses.replace(BASE, mac_lanes=8))).compute_busy_cycles)


def test_validation():
    import pytest
    with pytest.raises(ValueError):
        dataclasses.replace(BASE, mac_lanes=0).validate()
    with pytest.raises(ValueError):
        dataclasses.replace(BASE, sram_read_bytes_per_cycle=-1).validate()
