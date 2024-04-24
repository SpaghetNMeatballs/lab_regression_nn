from lipo_loader import load_lipo
from lipo_dataclass import LipoParams
import numpy as np
import scipy.stats as stats


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
    print(f"Оценки b: {[np.round(i,4) for i in b]}")
    _y = np.dot(parameter_matrix, b)
    e = y - _y
    print(f"МНК остатки e: {[np.round(i,4) for i in e]}")
    n = len(y)
    p = len(b)
    s2 = np.sum(e**2)/(n-p-1)
    ser = np.sqrt(s2)
    #print(f"ser: {[np.round(i,4) for i in ser]}")
    r2notcentralised = 1 - np.dot(np.transpose(e), e) / (np.dot(np.transpose(y), y))
    print(f"Коэффициент детерминации R2 = {r2notcentralised:4f}")
    t_statistics = b / np.sqrt(s2)
    df = n - p - 1
    confidence_level = 0.95
    alpha = 1 - confidence_level
    lower_bounds = b - stats.t.ppf(1 - alpha / 2, df) * np.sqrt(s2)
    upper_bounds = b + stats.t.ppf(1 - alpha / 2, df) * np.sqrt(s2)
    print("ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ")
    print(" t : ")
    for i in range(len(b)):
        print(
            " [ ",
            lower_bounds[i],
            " : ",
            upper_bounds[i],
            " ] ",
            b[i],
            " Входит " if upper_bounds[i] >= b[i] >= lower_bounds[i] else "Выходит",
        )


if __name__ == "__main__":
    main()
