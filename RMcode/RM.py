import numpy as np
import math
import itertools
from itertools import combinations
from itertools import islice
from functools import lru_cache
small_values_cache = {}
@lru_cache(None)
def binomial_cached(m, i):
    return math.comb(m, i)

def sum_binomial(m, r):
    total = 0
    for i in range(r + 1):
        binom = binomial_cached(m, i)
        total += binom
    return total
def replace_elements(arr, target, replacement):
    return [replacement if x == target else x for x in arr]
def xor_all_elements(arr):
    result = 0
    for num in arr:
        result ^= num
    return result
def create_string(A, B, a_coordinates):
    A = [str(a) for a in A]
    B = [str(b) for b in B]
    result = [''] * (len(A) + len(B))
    for i, pos in enumerate(a_coordinates):
        result[pos] = A[i]

    b_index = 0
    for i in range(len(result)):
        if result[i] == '':
            result[i] = B[b_index]
            b_index += 1

    return ''.join(result)
@lru_cache(None)
def get_multipliers(num_multipliers, num_x, idx):
    if (math.comb(num_x, num_multipliers)<5000):
        return get_multipliers_for_small_values(num_multipliers, num_x, idx)
    else:
        return get_multipliers_for_huge_values(num_multipliers, num_x, idx)

def get_multipliers_for_huge_values(num_multipliers, num_x, idx):
    try:
        selected_combination = next(islice(combinations(range(0, num_x), num_multipliers), idx, None))
    except StopIteration:
        raise IndexError("Индекс выходит за пределы доступных комбинаций")
    return selected_combination


def get_multipliers_for_small_values(num_multipliers, num_x, idx):
    cache_key = (num_multipliers, num_x)
    if cache_key not in small_values_cache:
        small_values_cache[cache_key] = list(combinations(range(0, num_x), num_multipliers))
    index_combinations = small_values_cache[cache_key]
    if idx < 0 or idx >= len(index_combinations):
        raise IndexError("Индекс выходит за пределы доступных комбинаций")
    return index_combinations[idx]
def fill_bit_array(n):
    num_rows = 2 ** n
    array = [[0] * num_rows for _ in range(n)]
    for i in range(num_rows):
        bit_representation = format(i, f'0{n}b')
        for j in range(n):
            array[j][i] = int(bit_representation[j])
    return array

def Creation_G1(m):
    return (fill_bit_array(m))
def Generation_Gr(g1, r):
    g1 = np.atleast_2d(g1)
    combinations = itertools.combinations(g1, r)
    res= []
    for combo in combinations:
        bitwise_product = np.bitwise_and.reduce(combo)
        res.append(bitwise_product)

    return np.matrix(res)
def MatrixConcat(m1,m2):
    return np.vstack((m1,  m2))


def matrix_generator(m, r):
    n=2**m
    k=sum_binomial(m, r)
    g_0=[1]*n
    g_0=np.array(g_0)
    g_0 = g_0.reshape(1, -1)
    g_1=Creation_G1(m)
    g_1 = np.array(g_1)
    g_r=Generation_Gr(g_1, r)
    res=np.vstack((g_0,g_1))
    for i in range(2,r+1):
        res=MatrixConcat(res,Generation_Gr(g_1,i))
    return res

def find_most_common_bit(z):
    count_zeros = z.count(0)
    count_ones = z.count(1)
    if count_zeros > count_ones:
        return 0
    else:
        return 1



class RM:
    def __init__(self, m, r):
        self.m = m
        self.r = r
        self.n = 2 ** m
        self.k = sum_binomial(m, r)
        self.d = 2 ** (self.m - self.r)
        self.MistakesCount = self.d / 2 - 1
        self.ErasesCount = self.d - 1
        self.matrix_cache = {}
        self.g1 =Creation_G1(m)
        self.gr_cache = {}

    def comb(self, n, k):
        return math.comb(n, k)

    def GetErasesCount(self):
        return self.ErasesCount

    def GetMistakesCount(self):
        return self.MistakesCount

    def find_degree_block_lens(self):
        return [self.comb(self.m, i) for i in range(self.r, -1, -1)]

    def get_matrix(self, m, r):
        if (m, r) not in self.matrix_cache:
            self.matrix_cache[(m, r)] = matrix_generator(m, r)
        return self.matrix_cache[(m, r)]


    def get_gr(self, r):
        if r not in self.gr_cache:
            self.gr_cache[r] = Generation_Gr(self.g1, r)
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
                resarr = []
                binary_str = bin(i)[2:].zfill(self.m - len(coords_to_fill))

                for j in range(2 ** degree):
                    binary_str2 = bin(j)[2:].zfill(len(coords_to_fill))
                    pos = int(create_string(binary_str2, binary_str, coords_to_fill), 2)
                    resarr.append(encoded_word[pos])

                x_variants.append(xor_all_elements(resarr))

            result1 += str(find_most_common_bit(x_variants))
        return result1

    def decode2(self, messandmis):
        z = messandmis.copy()
        res = []
        for i in range(self.r, 0, -1):
            mi = list(map(int, self.decode_highest_degree_block(self.comb(self.m, i), z, i)))
            res = mi + res
            z = (z - (np.array(mi) @ self.get_gr( i)) % 2) % 2
            z = z.A1
        res = [1 if sum(z) > 2 ** (self.m - 1) else 0] + res
        return res

    def Decode(self, emess):  # even with erases
        ones = replace_elements(emess.copy(), 3, 1)
        zeros = replace_elements(emess.copy(), 3, 0)
        o = self.decode2(ones)
        o_var = self.encode(o)
        z = self.decode2(zeros)
        z_var = self.encode(z)
        diff_count = lambda a: sum(x != y for x, y in zip(emess, a))
        return o if diff_count(o_var) <= diff_count(z_var) else z
