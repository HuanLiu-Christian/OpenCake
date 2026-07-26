# MLIR-AIR integration

This image provides the official `air-opt` binary from the MLIR-AIR Linux
wheel. OpenCake streams MLIR through `docker exec -i`, so no host workspace is
mounted into the container.

```bash
docker build --platform linux/amd64 -t opencake-mlir-air integrations/mlir_air_opencake
docker run -d --name opencake-air-dev opencake-mlir-air
```

The GEMM driver invokes these official transformations:

1. Transform Dialect one-shot bufferization.
2. `scf.forall` to `scf.parallel` conversion.
3. MLIR-AIR `--air-par-to-herd=depth=-1` lowering.
4. AIR channel, DMA, PE-local buffer, and K-stream materialization.
5. MLIR-AIR `--air-dependency` and `--air-dependency-canonicalize`.
6. Async AIR assembly emission for the `air-to-opencake` adapter.

The adapter retains `air.herd`, the nested K loop, tiled `linalg.matmul`,
`air.channel.put/get`, DMA operations, and async dependency tokens. OpenCake
then maps logical AIR channels onto wafer-specific color IDs and physical
routes, assigns SRAM addresses, and lowers the dependency graph to the timing
and Chisel backends before emitting Spatial IR.
