"""Independent-operand extension arithmetic via alpha cuts."""
from dataclasses import dataclass
import numpy as np
from .model import AlphaCutFuzzy, ComputationConfig, FuzzyValue


def arithmetic(a: FuzzyValue, b: FuzzyValue, operation: str,
               config: ComputationConfig = ComputationConfig()) -> AlphaCutFuzzy:
    x, y = a.to_alpha_cuts(config), b.to_alpha_cuts(config)
    if operation == "+":
        lo, hi = x.left + y.left, x.right + y.right
    elif operation == "-":
        lo, hi = x.left - y.right, x.right - y.left
    elif operation in ("*", "/"):
        if operation == "/" and np.any((y.left <= 0) & (y.right >= 0)):
            raise ZeroDivisionError("Срез знаменателя содержит ноль: деление не определено.")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            products = np.array([u * v if operation == "*" else u / v
                                 for u in (x.left, x.right) for v in (y.left, y.right)])
        lo, hi = products.min(axis=0), products.max(axis=0)
    else:
        raise ValueError("Неизвестная операция. Допустимы +, -, *, /.")
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError("Переполнение: уменьшите параметры величин.")
    return AlphaCutFuzzy(x.alpha, lo, hi, f"({a.label or 'A'} {operation} {b.label or 'B'})")


def add(a, b, config=ComputationConfig()): return arithmetic(a, b, "+", config)
def subtract(a, b, config=ComputationConfig()): return arithmetic(a, b, "-", config)
def multiply(a, b, config=ComputationConfig()): return arithmetic(a, b, "*", config)
def divide(a, b, config=ComputationConfig()): return arithmetic(a, b, "/", config)


def rank(a: FuzzyValue, config: ComputationConfig = ComputationConfig()) -> float:
    cuts = a.to_alpha_cuts(config)
    return float(np.trapezoid((cuts.left + cuts.right) / 2, cuts.alpha) / (1 - config.epsilon))


def distance(a: FuzzyValue, b: FuzzyValue, config: ComputationConfig = ComputationConfig()) -> float:
    x, y = a.to_alpha_cuts(config), b.to_alpha_cuts(config)
    delta = (x.left - y.left) ** 2 + (x.right - y.right) ** 2
    return float(np.sqrt(np.trapezoid(delta, x.alpha) / (2 * (1 - config.epsilon))))


@dataclass(frozen=True)
class Comparison:
    relation: str
    rank_a: float
    rank_b: float
    distance: float
    overlap_05: tuple[float, float] | None


def compare(a: FuzzyValue, b: FuzzyValue, config: ComputationConfig = ComputationConfig()) -> Comparison:
    ra, rb = rank(a, config), rank(b, config)
    tolerance = 1e-9 * max(1, abs(ra), abs(rb))
    relation = "=" if abs(ra - rb) <= tolerance else ("<" if ra < rb else ">")
    al, ar = a.alpha_cut(0.5)
    bl, br = b.alpha_cut(0.5)
    lo, hi = max(al, bl), min(ar, br)
    return Comparison(relation, ra, rb, distance(a, b, config), (lo, hi) if lo <= hi else None)
