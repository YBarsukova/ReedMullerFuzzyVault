from __future__ import annotations
import math
from functools import lru_cache
from itertools import combinations, islice
from typing import Dict, List, Tuple

import numpy as np

try:
    from numba import njit

    _HAS_NUMBA = True
except ModuleNotFoundError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def _decor(f):
            return f
        return _decor

@lru_cache(maxsize=None)
def _binom(m: int, k: int) -> int:
    return math.comb(m, k)

_small_values_cache: Dict[Tuple[int, int], List[Tuple[int, ...]]] = {}
_matrix_cache: Dict[Tuple[int, int], np.ndarray] = {}
_gr_cache: Dict[Tuple[int, int], np.ndarray] = {}
_combo_cache: Dict[Tuple[int, int], List[Tuple[int, ...]]] = {}
def _xor_all(arr: List[int]) -> int:
    x = 0
    for v in arr:
        x ^= v
    return x

def _majority(arr: List[int]) -> int:
    zeros = arr.count(0)
    return 0 if zeros > len(arr) // 2 else 1

def _all_combos(m: int, degree: int) -> List[Tuple[int, ...]]:
    key = (m, degree)
    if key not in _combo_cache:
        _combo_cache[key] = list(combinations(range(m), degree))
    return _combo_cache[key]

def _generation_g1(m: int) -> np.ndarray:
    n = 1 << m
    arr = np.zeros((m, n), dtype=np.uint8)
    for i in range(m):
        period = 1 << (m - i - 1)
        arr[i] = np.tile(np.concatenate([
            np.zeros(period, dtype=np.uint8),
            np.ones(period, dtype=np.uint8)
        ]), 1 << i)
    return arr

def _generation_gr(m: int, degree: int) -> np.ndarray:
    g1 = _generation_g1(m)
    combos = _all_combos(m, degree)
    res = np.empty((len(combos), 1 << m), dtype=np.uint8)
    for idx, comb in enumerate(combos):
        res[idx] = np.bitwise_and.reduce(g1[list(comb)], axis=0)
    return res

def _get_gr(m: int, degree: int) -> np.ndarray:
    key = (m, degree)
    if key not in _gr_cache:
        _gr_cache[key] = _generation_gr(m, degree)
    return _gr_cache[key]

def _get_matrix(m: int, r: int) -> np.ndarray:
    key = (m, r)
    if key not in _matrix_cache:
        rows = [np.ones((1, 1 << m), dtype=np.uint8)]
        rows.append(_generation_g1(m))
        for deg in range(2, r + 1):
            rows.append(_get_gr(m, deg))
        _matrix_cache[key] = np.vstack(rows)
    return _matrix_cache[key]

if _HAS_NUMBA:

    @njit(cache=True, fastmath=True)
    def _decode_block_jit(
        encoded: np.ndarray,
        m: int,
        degree: int,
        combos: np.ndarray,
        pow2: np.ndarray
    ) -> np.ndarray:
        num_variants = 1 << (m - degree)
        deg_variants = 1 << degree
        n_rows = combos.shape[0]
        result = np.empty(n_rows, dtype=np.uint8)
        for row in range(n_rows):
            coords = combos[row]
            zeros = 0
            ones = 0
            for free in range(num_variants):
                parity = 0
                for var_bits in range(deg_variants):
                    pos = 0
                    c_idx = 0
                    f_idx = 0
                    for bit in range(m):
                        in_coords = False
                        for cc in range(degree):
                            if coords[cc] == bit:
                                in_coords = True
                                break
                        if in_coords:
                            bit_val = (var_bits >> (degree - 1 - c_idx)) & 1
                            c_idx += 1
                        else:
                            bit_val = (free >> (m - degree - 1 - f_idx)) & 1
                            f_idx += 1
                        if bit_val:
                            pos |= pow2[bit]
                    parity ^= encoded[pos]
                if parity:
                    ones += 1
                else:
                    zeros += 1
            result[row] = 0 if zeros > ones else 1  # tie→1
        return result
else:
    def _decode_block_jit(*args, **kwargs):
        raise NotImplementedError
class CombinedRM:
    def __init__(self, m: int, r: int) -> None:
        if not (0 <= r <= m):
            raise ValueError("0 ≤ r ≤ m is required")
        self.m, self.r = m, r
        self.n = 1 << m
        self.k = sum(_binom(m, i) for i in range(r + 1))
        self._pow2 = np.array([1 << (m - bit - 1) for bit in range(m)], dtype=np.uint32)
        self._warm()
    def _warm(self) -> None:
        _get_matrix(self.m, self.r)
        for d in range(1, self.r + 1):
            _get_gr(self.m, d)
            _all_combos(self.m, d)
    def encode(self, message: List[int] | np.ndarray) -> np.ndarray:
        msg = np.asarray(message, dtype=np.uint8).flatten()
        if msg.size != self.k:
            raise ValueError(f"Message length must be {self.k}")
        G = _get_matrix(self.m, self.r)
        return (msg @ G) % 2
    def _decode_highest_degree_block(self, block_len: int, encoded: np.ndarray, degree: int) -> List[int]:
        if _HAS_NUMBA:
            combos_arr = np.array(_all_combos(self.m, degree), dtype=np.int8)
            coeffs = _decode_block_jit(encoded, self.m, degree, combos_arr, self._pow2)
            return coeffs.tolist()
        result: List[int] = []
        m = self.m
        for k in range(block_len):
            coords = _get_multipliers(degree, m, k)
            if not coords:
                result.append(_majority(encoded.tolist()))
                continue

            x_variants: List[int] = []
            for i in range(1 << (m - degree)):
                res_arr: List[int] = []
                for j in range(1 << degree):
                    bits = [0] * m
                    for idx_c, pos in enumerate(coords):
                        bits[pos] = (j >> (degree - idx_c - 1)) & 1
                    other_mask = i
                    for idx in range(m):
                        if idx not in coords:
                            bits[idx] = (other_mask >> (m - degree - 1)) & 1
                            other_mask >>= 1
                    pos_int = 0
                    for idx in range(m):
                        pos_int = (pos_int << 1) | bits[idx]
                    res_arr.append(int(encoded[pos_int]))
                x_variants.append(_xor_all(res_arr))
            result.append(_majority(x_variants))
        return result
    def decode_without_erasures(self, word: np.ndarray | List[int]) -> List[int]:
        z = np.asarray(word, dtype=np.uint8).flatten().copy()
        res: List[int] = []
        for deg in range(self.r, 0, -1):
            mi = self._decode_highest_degree_block(_binom(self.m, deg), z, deg)
            res = mi + res
            gr = _get_gr(self.m, deg)
            z = (z - (np.array(mi, dtype=np.uint8) @ gr) % 2) % 2
        res = [_majority(z.tolist())] + res
        return res
    def decode(self, word: np.ndarray | List[int]) -> List[int]:
        w = np.asarray(word, dtype=np.uint8).flatten()
        if w.size != self.n:
            raise ValueError(f"Word length must be {self.n}")
        return self.decode_without_erasures(w)
def _get_multipliers(num_mult: int, num_x: int, idx: int) -> Tuple[int, ...]:
    if _binom(num_x, num_mult) < 5000:
        key = (num_mult, num_x)
        if key not in _small_values_cache:
            _small_values_cache[key] = list(combinations(range(num_x), num_mult))
        return _small_values_cache[key][idx]
    try:
        return next(islice(combinations(range(num_x), num_mult), idx, None))
    except StopIteration as exc:
        raise IndexError("Index out of bounds for combinations") from exc
