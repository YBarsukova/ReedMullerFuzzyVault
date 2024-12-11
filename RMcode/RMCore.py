import numpy as np
from itertools import product
import RM
def combine_messages(u_hat, v_hat):
    decoded_message = np.concatenate((v_hat, u_hat))
    return decoded_message

class RMCore():
    def __init__(self, m,r):
        self.code=RM.RM(m,r)
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
    def decode_recursed_splited(self, message):
        assert(self.code.r==2)
        self.data=self.map_creation(self.code.m)

    def fullfill_maps(self, length):
        a = [''.join(bits) for bits in product('01', repeat=length)]
        code = RM.RM(length, 1)
        return {key: code.encode(key) for key in a}

    def map_creation(self, m):
        res = {i: None for i in range(4, m)}
        for i in range(4, m):
            res[i] = self.fullfill_maps(i)
        return res

