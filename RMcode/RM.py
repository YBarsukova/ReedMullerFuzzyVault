import numpy as np
import math
import itertools
from itertools import combinations
def sum_binomial(m, r):
    total = 0
    for i in range(r + 1):
        binom = math.comb(m, i)
        total += binom
    return total
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
def get_multipliers(num_multipliers, num_x, idx):
    index_combinations = list(combinations(range(1, num_x + 1), num_multipliers))
    if idx < 0 or idx >= len(index_combinations):
        raise IndexError("Индекс выходит за пределы доступных комбинаций")
    return tuple(x - 1 for x in index_combinations[idx])
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
        self.n = 2**m
        self.k =sum_binomial(m, r)

    def comb(self, n, k):
        """Функция для вычисления комбинаций C(n, k) = n! / (k! * (n-k)!)"""
        return math.comb(n, k)

    def find_degree_block_lens(self):
        result = []
        i = self.k
        while i > 0:
            # Вычисляем c(i) как комбинацию C из M по (R + len(result))
            c_i = self.comb(self.m, self.r + len(result))
            result.insert(0, c_i)  # Вставляем подмассив в начало результата
            i -= c_i  # Двигаемся на c(i) элементов назад
        result.reverse()
        return result

    def encode(self, message):
        result = np.dot(np.array(message), matrix_generator(self.m, self.r))
        ##result = np.dot(np.array(message), C)
        return np.array(result % 2).flatten()

    def decode_highest_degree_block(self, block_len, encoded_word, degree):
        result1=""
        for k in range(0,block_len):
            coords_to_fill=get_multipliers(degree, self.m, k)
            if len(coords_to_fill)==0:
                result1 += str(find_most_common_bit(encoded_word.tolist()))
                continue
            x_variants = []

            for i in range (0, 2**(self.m-degree)): ##b = 0..0, ... 1..1
                resarr=[]
                binary_str = bin(i)[2:]
                binary_str=binary_str.zfill(self.m-len(coords_to_fill))

                for j in range (0, 2**degree):
                    binary_str2 = bin(j)[2:]
                    binary_str2 = binary_str2.zfill(len(coords_to_fill))
                    pos = int(create_string(binary_str2, binary_str, coords_to_fill), 2)
                    resarr.append(encoded_word[pos])

                x_variants.append(xor_all_elements(resarr))

            result1+=str(find_most_common_bit(x_variants))
        return result1[::-1]

    def decode(self, messandmis):
        result1=""
        arr=self.find_degree_block_lens()
        newq=messandmis
        prev=newq
        for i in range(0, self.r+1):
            if i>0:
                if i==self.r:
                    temp = list(prev)
                    temp.reverse()
                    pr = np.matrix([int(q) for q in temp])

                    vichitaemoe = np.dot(pr, np.matrix(Creation_G1(self.m)))
                    newq = np.array(np.subtract(messandmis, vichitaemoe) % 2).flatten()
                else:
                    temp = list(result1)
                    temp.reverse()
                    pr=np.matrix([int(q) for q in temp])
                    vichitaemoe = np.dot(pr, Generation_Gr(Creation_G1(self.m),self.r))%2
                    newq=np.array(np.subtract(messandmis, vichitaemoe)%2).flatten()
                    messandmis = newq
                    prev = self.decode_highest_degree_block(arr[i], newq, self.r - i)
            result1+=self.decode_highest_degree_block(arr[i], newq, self.r - i)
        return result1[::-1]

