FROM python:3.12-slim

RUN python -m pip install --no-cache-dir \
      'mlir_air==0.0.1.2026071905+0a2dc35' \
      -f https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels \
    && air-opt --version

CMD ["sleep", "infinity"]
