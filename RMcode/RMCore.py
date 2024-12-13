import numpy as np
from itertools import product
from Levenshtein import distance as levenshtein_distance
import RM
from RMcode.RM import sum_binomial

def levenshtein_distance1(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]
    for i in range(len_s1 + 1):
        dp[i][0] = i
    for j in range(len_s2 + 1):
        dp[0][j] = j
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1]
                )
    return dp[len_s1][len_s2]

def combine_messages(u_hat, v_hat):
    decoded_message = np.concatenate((v_hat, u_hat))
    return decoded_message

class RMCore():
    def __init__(self, m,r):
        self.code=RM.RM(m,r)
        self.data= self.initialize_data(3, m)
    def encode(self, message):
        return self.code.encode(message)
    def decode_classic(self, message):
        return self.code.decode(message)
    def decode_classic_splited(self, y):
        n = len(y)
        mid = n // 2
        y_L = y[:mid]
        y_R = y[mid:]
        s = np.bitwise_xor(y_L, y_R)
        c1 = RM.RM(self.code.m - 1, self.code.r - 1)
        v_hat = c1.decode(s)
        v_hat_codeword = c1.encode(v_hat)
        y_L_corrected1 = np.bitwise_xor(y_L, v_hat_codeword)
        y_L_corrected2 = y_L.copy()
        c2 = RM.RM(self.code.m - 1, self.code.r)
        u_hat1 = c2.decode(y_L_corrected1)
        u_hat2 = c2.decode(y_L_corrected2)
        u_hat_codeword1 = c2.encode(u_hat1)
        u_hat_codeword2 = c2.encode(u_hat2)
        codeword1_left = u_hat_codeword1
        codeword1_right = np.bitwise_xor(u_hat_codeword1, v_hat_codeword)
        codeword1 = np.concatenate((codeword1_left, codeword1_right))
        codeword2_left = u_hat_codeword2
        codeword2_right = np.bitwise_xor(u_hat_codeword2, v_hat_codeword)
        codeword2 = np.concatenate((codeword2_left, codeword2_right))
        diff1 = np.sum(y != codeword1)
        diff2 = np.sum(y != codeword2)
        if diff1 <= diff2:
            decoded_message = combine_messages(u_hat1, v_hat)
        else:
            decoded_message = combine_messages(u_hat2, v_hat)
        return decoded_message

    def decode_recursed_splited(self, y):

        assert self.code.r == 2
        n = len(y)
        mid = n // 2
        y_L = y[:mid]
        y_R = y[mid:]
        s = np.bitwise_xor(y_L, y_R)
        # decode_map = self.data[self.code.m - 1]
        # v_hat = self.decode_using_map(s, decode_map)
        # v_hat_codeword = decode_map[tuple(v_hat)]
        c1 = RM.RM(self.code.m - 1, self.code.r - 1)
        v_hat = c1.decode(s)
        v_hat_codeword = c1.encode(v_hat)
        c2 = RM.RM(self.code.m - 1, self.code.r)
        y_L_corrected1 = np.bitwise_xor(y_L, v_hat_codeword)
        y_L_corrected2 = y_L.copy()
        c2 = RM.RM(self.code.m - 1, self.code.r)
        u_hat1 = c2.decode(y_L_corrected1)
        u_hat2 = c2.decode(y_L_corrected2)
        u_hat_codeword1 = c2.encode(u_hat1)
        u_hat_codeword2 = c2.encode(u_hat2)
        codeword1_left = u_hat_codeword1
        codeword1_right = np.bitwise_xor(u_hat_codeword1, v_hat_codeword)
        codeword1 = np.concatenate((codeword1_left, codeword1_right))
        codeword2_left = u_hat_codeword2
        codeword2_right = np.bitwise_xor(u_hat_codeword2, v_hat_codeword)
        codeword2 = np.concatenate((codeword2_left, codeword2_right))
        diff1 = np.sum(y != codeword1)
        diff2 = np.sum(y != codeword2)
        if diff1 <= diff2:
            decoded_message = combine_messages(u_hat1, v_hat)
        else:
            decoded_message = combine_messages(u_hat2, v_hat)
        return decoded_message

    def map_creation(self, m):
        code = RM.RM(m, 1)
        bitstrings = [list(map(int, ''.join(bits))) for bits in product('01', repeat=sum_binomial(m,1))]
        result_map = {tuple(bitstring): code.encode(bitstring) for bitstring in bitstrings}
        return result_map

    def initialize_data(self, lower_bound, upper_bound):
        return {m: self.map_creation(m) for m in range(lower_bound, upper_bound)}

    def decode_using_map(self, bits, decode_map):
        bits_str = ''.join(map(str, bits))
        min_dis=len(bits_str)
        cur_key=None
        for key in decode_map.keys():
            key_str = ''.join(map(str, decode_map[key]))
            distance = levenshtein_distance1(bits_str, key_str)
            if distance < min_dis:
                min_dis = distance
                cur_key = key
        return cur_key

