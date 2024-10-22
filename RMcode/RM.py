import operator

import numpy as np


def generate_matrix():
    matrix = np.zeros((11, 16), dtype=int)
    matrix[0] = np.ones(16, dtype=int)
    matrix[1, 8:] = 1
    matrix[2, 4:8] = 1
    matrix[2, 12:16] = 1
    matrix[3, 2:4] = 1
    matrix[3, 6:8] = 1
    matrix[3, 10:12] = 1
    matrix[3, 14:16] = 1
    matrix[4, 1::2] = 1
    matrix[5, 12:] = 1
    matrix[6, 14:16] = 1
    matrix[6, 10:12] = 1
    matrix[7, 15] = 1
    matrix[7, 9] = 1
    matrix[7, 13] = 1
    matrix[7, 11] = 1
    matrix[8, 14::] = 1
    matrix[8, 6:8] = 1
    matrix[9, 15] = 1
    matrix[9, 13] = 1
    matrix[9, 5] = 1
    matrix[9, 7] = 1
    matrix[10, 3] = 1
    matrix[10, 7] = 1
    matrix[10, 11] = 1
    matrix[10, 15] = 1
    return matrix


def xor_array4(arr, zvec):
    z = [
        zvec[arr[0] - 1] ^ zvec[arr[1] - 1] ^ zvec[arr[2] - 1] ^ zvec[arr[3] - 1],
        zvec[arr[4] - 1] ^ zvec[arr[5] - 1] ^ zvec[arr[6] - 1] ^ zvec[arr[7] - 1],
        zvec[arr[8] - 1] ^ zvec[arr[9] - 1] ^ zvec[arr[10] - 1] ^ zvec[arr[11] - 1],
        zvec[arr[12] - 1] ^ zvec[arr[13] - 1] ^ zvec[arr[14] - 1] ^ zvec[arr[15] - 1],
    ]
    return z


def xor_array8(arr, zvec):
    z = [
        zvec[arr[0] - 1] ^ zvec[arr[1] - 1],
        zvec[arr[2] - 1] ^ zvec[arr[3] - 1],
        zvec[arr[4] - 1] ^ zvec[arr[5] - 1],
        zvec[arr[6] - 1] ^ zvec[arr[7] - 1],
        zvec[arr[8] - 1] ^ zvec[arr[9] - 1],
        zvec[arr[10] - 1] ^ zvec[arr[11] - 1],
        zvec[arr[12] - 1] ^ zvec[arr[13] - 1],
        zvec[arr[14] - 1] ^ zvec[arr[15] - 1],
    ]
    return z


def find_most_common_bit(z):
    count_zeros = z.count(0)
    count_ones = z.count(1)
    if count_zeros > count_ones:
        return 0
    else:
        return 1


class RM:
    def init(self, m, r, n, k):
        self.m = m
        self.r = r
        self.n = n
        self.k = k

    def Encoding42(self, message):
        result = np.dot(np.array(message), generate_matrix())
        return result % 2

    def Decoding42(self, messandmis):
        z4 = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            [1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16],
            [1, 3, 5, 7, 2, 4, 6, 8, 9, 11, 13, 15, 10, 12, 14, 16],
            [1, 2, 9, 10, 3, 4, 11, 12, 5, 6, 13, 14, 7, 8, 15, 16],
            [1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16],
            [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16]
        ]
        z8 = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            [1, 3, 2, 4, 5, 7, 6, 8, 9, 11, 10, 12, 13, 15, 14, 16],
            [1, 5, 2, 6, 3, 7, 4, 8, 9, 13, 10, 14, 11, 15, 12, 16],
            [1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15, 8, 16]
        ]

        zvec = messandmis
        x = []

        for zi in z4:
            z = xor_array4(zi, zvec)
            most_common_bit = find_most_common_bit(z)
            x.append(most_common_bit)

        zvec = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1]

        for zi in z8:
            z = xor_array8(zi, zvec)
            most_common_bit = find_most_common_bit(z)
            x.append(most_common_bit)

        zvec = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        x.append(find_most_common_bit(zvec))
        x.reverse()
        return x

