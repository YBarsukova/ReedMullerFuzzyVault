from copy import copy

import numpy as np
from numba import typeof

import RM
import random


def random_set(count, min, max):
    unique_numbers = set()
    while len(unique_numbers) < count:
        number = random.randint(min, max)
        unique_numbers.add(number)
    return unique_numbers


def combine_messages(u_hat, v_hat):
    # Получаем длину сообщений u_hat и v_hat
    k_u = len(u_hat)
    k_v = len(v_hat)
    # Предположим, что исходное сообщение состоит из объединения u_hat и v_hat
    decoded_message = np.concatenate((v_hat, u_hat))
    return decoded_message



def split_array(arr):
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    return left, right
def missing_numbers_in_range(min_value, max_value, us_set):
    full_set = set(range(min_value, max_value + 1))
    missing_numbers = full_set - us_set
    return missing_numbers

class Vault:
    def __init__(self,m,r):
        self.code=RM.RM(m,r)
        self.attempts_count=3

    def filling_trash(self, enc, count):
        rs=random_set(count, 0,  self.code.n-1)
        self.kee=missing_numbers_in_range(0, self.code.n-1, rs)
        for ind in rs:
            enc[ind]=enc[ind]^1
        return enc

    def filling_erases(self, vault, ints):
        for m in missing_numbers_in_range(0, self.code.n-1, ints):
            vault[m]=3
        return vault

    def lock(self, data_input):
        self.secret=data_input
        enc=self.code.encode(data_input)
        self.vault=self.filling_trash(enc, self.code.erases_count-2)
        print(f"{self.kee} - yours password \n")

    def unlock(self, ints):
        if self.attempts_count>0:
            attempt=self.filling_erases(copy(self.vault), ints)
            #dec=self.code.decode(attempt)
            dec=list(self.splited_unlock(attempt))
            # print(dec, dec1)
            if (dec==self.secret):
                print("Unlocked!!!")
            else:
                print("try again")
                self.attempts_count-=1
                print(self.attempts_count)
        else:
            print("There is no attempts")

    def splited_unlock(self, y):
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


