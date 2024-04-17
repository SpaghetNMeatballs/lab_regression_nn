from lipo_loader import load_lipo
from lipo_dataclass import LipoParams
import numpy as np


def calc_b(
    x: np.array,
    y: np.array,
) -> np.array:
    x_t = np.transpose(x)
    first = np.dot(x_t, x)
    second = np.linalg.inv(first)
    return np.dot(np.dot(second, x_t), y)


def main():
    db = load_lipo()
    parameter_matrix = np.array([np.array(i.parameters) for i in db])
    y = np.array([i.logP for i in db])
    b = calc_b(parameter_matrix, y)
    print(f"b: {[np.round(i,4) for i in b]}")
    _y = np.dot(parameter_matrix, b)
    e = y - _y
    print(f"e: {[np.round(i,4) for i in e]}")
    x_t = np.transpose(parameter_matrix)
    xtx = np.dot(x_t, parameter_matrix)
    xtxinv = np.linalg.inv(xtx)
    P = np.dot(np.dot(parameter_matrix, xtxinv), x_t)
    M = 1 - P
    ssr = np.transpose(e) * e
    n = len(y)
    k = len(b)
    s2 = ssr / (n - k)
    se = [np.sqrt(np.dot(s2, xtxinv[i][i])) for i in range(len(xtxinv))]
    t = b / np.sqrt(s2)
    ser = np.sqrt(s2)
    print(f"ser: {[np.round(i,4) for i in ser]}")
    r2notcentralised = 1 - np.dot(np.transpose(e), e) / (np.dot(np.transpose(y), y))
    print(f"R2 = {r2notcentralised:4f}")


if __name__ == "__main__":
    main()
