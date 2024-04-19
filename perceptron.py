from lipo_loader import load_lipo
from random import uniform, randint


def check_set(set: list[tuple, ...]) -> bool:
    len_specimen_x = len(set[0][0])
    for specimen in set:
        if len_specimen_x != len(specimen[0]):
            return False
    return True


class Perceptron:
    def __init__(
        self,
        delta: float,
        eps: float,
    ):
        self.eps = eps
        self.delta = delta
        self.j = 0

    def train_reverse_spread(
        self,
        train_set: list[tuple, ...],
        hidden_layer_count: int,
    ):
        if not check_set(train_set):
            raise Exception("Set failed verification")
        m = len(train_set[0][0])
        l = hidden_layer_count
        w_hidden = [
            [uniform(-(m**-1 * (l + 1) ** -1), m**-1 * (l + 1) ** -1) for j in range(m)]
            for i in range(l)
        ]
        w_outer = [0 for i in range(l)]
