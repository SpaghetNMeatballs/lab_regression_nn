from dataclasses import dataclass


@dataclass
class LipoParams:
    name: str
    logP: float
    parameters: list[int]
