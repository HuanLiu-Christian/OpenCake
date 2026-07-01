"""Pure functional kernels used by workload-level dataflow models."""


def gemv(
    matrix: list[float],
    vector: list[float],
    bias: list[float],
    rows: int,
    cols: int,
) -> list[float]:
    if len(matrix) != rows * cols:
        raise ValueError("matrix shape does not match rows and cols")
    if len(vector) != cols:
        raise ValueError("vector length does not match cols")
    if len(bias) != rows:
        raise ValueError("bias length does not match rows")

    output = []
    for row in range(rows):
        acc = 0.0
        for col in range(cols):
            acc += matrix[row * cols + col] * vector[col]
        output.append(acc + bias[row])
    return output


def gemv_ops(rows: int, cols: int, ops_per_mac: int = 2) -> int:
    return rows * cols * ops_per_mac


def gemm(
    left: list[float],
    right: list[float],
    m: int,
    n: int,
    k: int,
) -> list[float]:
    if len(left) != m * k:
        raise ValueError("left matrix shape does not match M x K")
    if len(right) != k * n:
        raise ValueError("right matrix shape does not match K x N")

    output = []
    for row in range(m):
        for col in range(n):
            acc = 0.0
            for kk in range(k):
                acc += left[row * k + kk] * right[kk * n + col]
            output.append(acc)
    return output


def gemm_ops(m: int, n: int, k: int, ops_per_mac: int = 2) -> int:
    return m * n * k * ops_per_mac


def vector_dot(
    left: list[float],
    right: list[float],
) -> float:
    if len(left) != len(right):
        raise ValueError("vector lengths must match")
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


def vector_dot_ops(length: int, ops_per_mac: int = 2) -> int:
    if length <= 0:
        raise ValueError("vector length must be positive")
    return length * ops_per_mac
