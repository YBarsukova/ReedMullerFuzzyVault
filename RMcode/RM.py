import numpy as np
import math
import itertools
from itertools import combinations, islice
from functools import lru_cache

small_values_cache = {}


@lru_cache(None)
def binomial_cached(m, i):
    return math.comb(m, i)


def sum_binomial(m, r):
    total = 0
    for i in range(r + 1):
        current = binomial_cached(m, i)
        total += current
    return total


def replace_elements(arr, target, replacement):
    return [replacement if x == target else x for x in arr]


def xor_all_elements(arr):
    result = 0
    for num in arr:
        result ^= num
    return result


def create_string(a_list, b_list, a_coordinates):
    a_list = [str(a) for a in a_list]
    b_list = [str(b) for b in b_list]
    result = [''] * (len(a_list) + len(b_list))
    for i, pos in enumerate(a_coordinates):
        result[pos] = a_list[i]

    b_index = 0
    for i in range(len(result)):
        if result[i] == '':
            result[i] = b_list[b_index]
            b_index += 1

    return ''.join(result)

import numpy as np

def create_array(a_array, b_array, a_coordinates):
    result = np.empty(len(a_array) + len(b_array), dtype=int)
    result.fill(-1)
    result[a_coordinates] = a_array
    b_index = 0
    for i in range(len(result)):
        if result[i] == -1:
            result[i] = b_array[b_index]
            b_index += 1
    return result

@lru_cache(None)
def get_multipliers(num_multipliers, num_x, idx):
    if binomial_cached(num_x, num_multipliers) < 5000:
        return get_multipliers_for_small_values(num_multipliers, num_x, idx)
    else:
        return get_multipliers_for_huge_values(num_multipliers, num_x, idx)


def get_multipliers_for_huge_values(num_multipliers, num_x, idx):
    try:
        selected_combination = next(islice(combinations(range(num_x), num_multipliers), idx, None))
    except StopIteration:
        raise IndexError("Index out of bounds for combinations")
    return selected_combination


def get_multipliers_for_small_values(num_multipliers, num_x, idx):
    cache_key = (num_multipliers, num_x)
    if cache_key not in small_values_cache:
        small_values_cache[cache_key] = list(combinations(range(num_x), num_multipliers))
    index_combinations = small_values_cache[cache_key]
    if idx < 0 or idx >= len(index_combinations):
        raise IndexError("Index out of bounds for combinations")
    return index_combinations[idx]


def fill_bit_array(n):
    num_rows = 2 ** n
    array = [[0] * num_rows for _ in range(n)]
    for i in range(num_rows):
        bit_representation = format(i, f'0{n}b')
        for j in range(n):
            array[j][i] = int(bit_representation[j])
    return array


def creation_g1(m):
    return fill_bit_array(m)


def generation_gr(g1, r):
    g1 = np.atleast_2d(g1)
    combs = itertools.combinations(g1, r)
    res = []
    for combo in combs:
        bitwise_product = np.bitwise_and.reduce(combo)
        res.append(bitwise_product)
    return np.matrix(res)


def matrix_concat(m1, m2):
    return np.vstack((m1, m2))


def matrix_generator(m, r):
    n = 2 ** m
    g_0 = [1] * n
    g_0 = np.array(g_0).reshape(1, -1)
    g_1 = creation_g1(m)
    g_1 = np.array(g_1)
    res = np.vstack((g_0, g_1))
    for i in range(2, r + 1):
        res = matrix_concat(res, generation_gr(g_1, i))
    return res


def find_most_common_bit(z):
    return 0 if z.count(0) > z.count(1) else 1


def comb(n, k):
    return binomial_cached(n, k)


class RM:
    def __init__(self, m, r):
        self.m = m
        self.r = r
        self.n = 2 ** m
        self.k = sum_binomial(m, r)
        self.d = 2 ** (self.m - self.r)
        self.mistakes_count = (self.d - 1)//2
        self.erases_count = self.d - 1
        self.matrix_cache = {}
        self.g1 = creation_g1(m)
        self.gr_cache = {}

    def get_erases_count(self):
        return self.erases_count

    def get_mistakes_count(self):
        return self.mistakes_count

    def find_degree_block_lens(self):
        return [comb(self.m, i) for i in range(self.r, -1, -1)]

    def get_matrix(self, m, r):
        if (m, r) not in self.matrix_cache:
            self.matrix_cache[(m, r)] = matrix_generator(m, r)
        return self.matrix_cache[(m, r)]

    def get_gr(self, r):
        if r not in self.gr_cache:
            self.gr_cache[r] = generation_gr(self.g1, r)
        return self.gr_cache[r]

    def encode(self, message):
        matrix = self.get_matrix(self.m, self.r)
        result = np.dot(np.array(message), matrix)
        return np.array(result % 2).flatten()

    def decode_highest_degree_block(self, block_len, encoded_word, degree):
        result1 = ""
        for k in range(block_len):
            coords_to_fill = get_multipliers(degree, self.m, k)
            if len(coords_to_fill) == 0:
                result1 += str(find_most_common_bit(encoded_word.tolist()))
                continue
            x_variants = []
            for i in range(2 ** (self.m - degree)):
                res_arr = []
                binary_str = bin(i)[2:].zfill(self.m - len(coords_to_fill))
                for j in range(2 ** degree):
                    binary_str2 = bin(j)[2:].zfill(len(coords_to_fill))
                    pos = int(create_string(binary_str2, binary_str, coords_to_fill), 2)
                    res_arr.append(encoded_word[pos])
                x_variants.append(xor_all_elements(res_arr))
            result1 += str(find_most_common_bit(x_variants))
        return result1

    def decode_without_erasures(self, mess_and_mis):
        z = mess_and_mis.copy()
        res = []
        for i in range(self.r, 0, -1):
            mi_str = self.decode_highest_degree_block(comb(self.m, i), z, i)
            mi = list(map(int, mi_str))
            res = mi + res
            z = (z - (np.array(mi) @ self.get_gr(i)) % 2) % 2
            z = z.A1
        total_sum = sum(z)
        res = [1 if total_sum > 2 ** (self.m - 1) else 0] + res
        return res

    def decode(self, emess):
        if np.count_nonzero(emess == 3) > 0:
            ones = replace_elements(emess.copy(), 3, 1)
            zeros = replace_elements(emess.copy(), 3, 0)
            o = self.decode_without_erasures(ones)
            o_var = self.encode(o)
            z = self.decode_without_erasures(zeros)
            z_var = self.encode(z)
            diff_count = lambda a: sum(x != y for x, y in zip(emess, a))
            return o if diff_count(o_var) <= diff_count(z_var) else z
        else:
            return self.decode_without_erasures(emess)