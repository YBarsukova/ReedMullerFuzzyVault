import random
from copy import copy

import numpy as np

from rm_code.reedmuller import ReedMuller


def random_set(count, min_value, max_value):
    """
    Generate a set of unique random integers in the inclusive range [min_value, max_value].
    """
    unique_numbers = set()
    while len(unique_numbers) < count:
        number = random.randint(min_value, max_value)
        unique_numbers.add(number)
    return unique_numbers


def combine_messages(u_hat, v_hat):
    """
    Combine decoded parts (u_hat, v_hat) into a single decoded message.
    """
    k_u = len(u_hat)
    k_v = len(v_hat)
    decoded_message = np.concatenate((v_hat, u_hat))
    return decoded_message


def split_array(arr):
    """
    Split an array into two halves (left, right).
    """
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    return left, right


def missing_numbers_in_range(min_value, max_value, us_set):
    """
    Return the set of numbers in [min_value, max_value] that are missing from us_set.
    """
    full_set = set(range(min_value, max_value + 1))
    missing_numbers = full_set - us_set
    return missing_numbers


class Vault:
    """
    A simple vault prototype based on Reed-Muller encoding/decoding with erasures.
    """

    def __init__(self, m, r):
        """
        Initialize the vault for a given Reed-Muller code RM(m, r).
        """
        self.key = None
        self.vault = None
        self.secret = None
        self.code = ReedMuller(m, r)
        self.attempts_count = 3

    def filling_trash(self, enc, count):
        """
        Flip 'count' random positions in the codeword and store the remaining indices as the key.
        """
        rs = random_set(count, 0, self.code.n - 1)
        self.key = missing_numbers_in_range(0, self.code.n - 1, rs)
        for ind in rs:
            enc[ind] = enc[ind] ^ 1
        return enc

    def filling_erases(self, vault, ints):
        """
        Mark positions not present in ints as erasures (value 3).
        """
        for m in missing_numbers_in_range(0, self.code.n - 1, ints):
            vault[m] = 3
        return vault

    def lock(self, data_input):
        """
        Lock the vault: encode the secret, inject noise, and return the key.
        """
        self.secret = data_input
        enc = self.code.encode(data_input)
        self.vault = self.filling_trash(enc, self.code.erases_count - 2)
        return self.key

    def unlock(self, ints):
        """
        Attempt to unlock the vault using the provided indices, limited by attempts_count.
        """
        if self.attempts_count > 0:
            attempt = self.filling_erases(copy(self.vault), ints)
            dec = list(self.splited_unlock(attempt))
            if dec == self.secret:
                print("Unlocked!!!")
            else:
                print("Try again.")
                self.attempts_count -= 1
                print(self.attempts_count)
        else:
            print("There are no attempts left.")

    def infinite_unlock(self, ints):
        """
        Attempt to unlock without decrementing attempts_count; returns True/False.
        """
        attempt = self.filling_erases(copy(self.vault), ints)
        dec = list(self.splited_unlock(attempt))
        if dec == self.secret:
            return True
        return False

    def splited_unlock(self, y):
        """
        Decode using a split (u, u+v) approach and choose the closer candidate codeword.
        """
        n = len(y)
        mid = n // 2
        y_l = y[:mid]
        y_r = y[mid:]

        s = np.bitwise_xor(y_l, y_r)

        c1 = ReedMuller(self.code.m - 1, self.code.r - 1)
        v_hat = c1.decode(s)
        v_hat_codeword = c1.encode(v_hat)

        y_l_corrected1 = np.bitwise_xor(y_l, v_hat_codeword)
        y_l_corrected2 = y_l.copy()

        c2 = ReedMuller(self.code.m - 1, self.code.r)
        u_hat1 = c2.decode(y_l_corrected1)
        u_hat2 = c2.decode(y_l_corrected2)

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