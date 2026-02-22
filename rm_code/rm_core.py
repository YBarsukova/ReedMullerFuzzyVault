import math
from itertools import product

import numpy as np

from rm_code.reedmuller import ReedMuller, sum_binomial


def levenshtein_distance(s1, s2):
    """
    Compute Levenshtein (edit) distance between two sequences.
    """
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
                    dp[i - 1][j - 1],
                )

    return dp[len_s1][len_s2]


def combine_messages(u_hat, v_hat):
    """
    Combine decoded parts (u_hat, v_hat) into a single decoded message.
    """
    decoded_message = np.concatenate((v_hat, u_hat))
    return decoded_message


class RMCore:
    """
    Wrapper around ReedMuller with split/recursive decoding helpers.
    """

    def __init__(self, m, r):
        """
        Initialize RMCore with base code RM(m, r) and internal code cache.
        """
        self.code = ReedMuller(m, r)
        # self.data = self.initialize_data(3, m)
        self.codes = {}

    def encode(self, message):
        """
        Encode message using the base ReedMuller instance.
        """
        return self.code.encode(message)

    def decode_classic(self, message):
        """
        Decode message using the base ReedMuller decoder.
        """
        return self.code.decode(message)

    def decode_classic_splited(self, y):
        """
        Classic (u, u+v) split decode:
        - decode v from XOR(left, right) using RM(m-1, r-1)
        - decode u from left (with and without v correction) using RM(m-1, r)
        - pick the closer candidate codeword
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

        codeword1 = np.concatenate(
            (u_hat_codeword1, np.bitwise_xor(u_hat_codeword1, v_hat_codeword))
        )
        codeword2 = np.concatenate(
            (u_hat_codeword2, np.bitwise_xor(u_hat_codeword2, v_hat_codeword))
        )

        diff1 = np.sum(y != codeword1)
        diff2 = np.sum(y != codeword2)

        if diff1 <= diff2:
            decoded_message = combine_messages(u_hat1, v_hat)
        else:
            decoded_message = combine_messages(u_hat2, v_hat)

        return decoded_message

    def decode_recursed_splited(self, y):
        """
        Recursive split decode variant for r == 2 using a precomputed map (self.data).
        """
        assert self.code.r == 2

        n = len(y)
        mid = n // 2
        y_l = y[:mid]
        y_r = y[mid:]

        s = np.bitwise_xor(y_l, y_r)

        decode_map = self.data[self.code.m - 1]
        v_hat = self.decode_using_map(s, decode_map)
        v_hat_codeword = decode_map[tuple(v_hat)]

        y_l_corrected1 = np.bitwise_xor(y_r, v_hat_codeword)
        y_l_corrected2 = y_l.copy()

        c2 = ReedMuller(self.code.m - 1, self.code.r)
        u_hat1 = c2.decode(y_l_corrected1)
        u_hat2 = c2.decode(y_l_corrected2)

        u_hat_codeword1 = c2.encode(u_hat1)
        u_hat_codeword2 = c2.encode(u_hat2)

        codeword1 = np.concatenate(
            (u_hat_codeword1, np.bitwise_xor(u_hat_codeword1, v_hat_codeword))
        )
        codeword2 = np.concatenate(
            (u_hat_codeword2, np.bitwise_xor(u_hat_codeword2, v_hat_codeword))
        )

        diff1 = np.sum(y != codeword1)
        diff2 = np.sum(y != codeword2)

        if diff1 <= diff2:
            return self.code.decode(codeword1)
        return self.code.decode(codeword2)

    def real_decode_first_degree(self, message):
        """
        Decode RM(m, 1) by Walsh-Hadamard style transform and return [sign | index_bits].
        """
        array = self.decode_first_degree(message)
        max_index = np.argmax(np.abs(array))
        sign = int(array[max_index] < 0)

        num_bits = int(math.log2(len(message)))
        binary_index = np.array(list(map(int, f"{max_index:0{num_bits}b}")), dtype=int)
        result = np.hstack(([sign], binary_index))

        return result

    def map_creation(self, m):
        """
        Build a lookup map from message bits to encoded codeword for RM(m, 1).
        """
        code = ReedMuller(m, 1)
        bitstrings = [
            list(map(int, "".join(bits)))
            for bits in product("01", repeat=sum_binomial(m, 1))
        ]
        result_map = {tuple(bitstring): code.encode(bitstring) for bitstring in bitstrings}
        return result_map

    def initialize_data(self, lower_bound, upper_bound):
        """
        Precompute maps for RM(m, 1) for m in [lower_bound, upper_bound).
        """
        return {m: self.map_creation(m) for m in range(lower_bound, upper_bound)}

    def decode_using_map(self, bits, decode_map):
        """
        Pick the closest codeword from decode_map using Levenshtein distance.
        Returns the selected message key.
        """
        bits_str = "".join(map(str, bits))
        min_dis = len(bits_str)
        cur_key = None

        for key in decode_map.keys():
            key_str = "".join(map(str, decode_map[key]))
            distance = levenshtein_distance(bits_str, key_str)
            if distance < min_dis:
                min_dis = distance
                cur_key = key

        return cur_key

    def get_code(self, m, r):
        """
        Get (or cache) a ReedMuller instance for RM(m, r).
        """
        if (m, r) not in self.codes:
            self.codes[(m, r)] = ReedMuller(m, r)
        return self.codes[(m, r)]

    def real_final_version_decode(self, message, depth=1):
        """
        Convenience wrapper to decode using final_version_decode with self.code parameters.
        """
        return self.final_version_decode(message, self.code.m, self.code.r, depth)

    def final_version_decode1(self, y, m, r, depth):
        """
        Older recursive decoder variant (kept as-is) specialized for r == 2.
        """
        if depth > 7:
            depth = 7

        assert self.code.r == 2

        n = len(y)
        mid = n // 2
        y_l = y[:mid]
        y_r = y[mid:]

        s = np.bitwise_xor(y_l, y_r)

        v_hat = self.real_decode_first_degree(s)
        v_hat_codeword = self.get_code(m - 1, r - 1).encode(v_hat)

        y_l_corrected1 = np.bitwise_xor(y_r, v_hat_codeword)
        y_l_corrected2 = y_l.copy()

        c2 = self.get_code(m - 1, r)
        if m > 4 and depth > 0 and r == m:
            u_hat1 = self.final_version_decode(y_l_corrected1, m - 1, r, depth - 1)
            u_hat2 = self.final_version_decode(y_l_corrected2, m - 1, r, depth - 1)
        else:
            u_hat1 = c2.decode(y_l_corrected1)
            u_hat2 = c2.decode(y_l_corrected2)

        u_hat_codeword1 = c2.encode(u_hat1)
        u_hat_codeword2 = c2.encode(u_hat2)

        codeword1 = np.concatenate(
            (u_hat_codeword1, np.bitwise_xor(u_hat_codeword1, v_hat_codeword))
        )
        codeword2 = np.concatenate(
            (u_hat_codeword2, np.bitwise_xor(u_hat_codeword2, v_hat_codeword))
        )

        diff1 = np.sum(y != codeword1)
        diff2 = np.sum(y != codeword2)

        cur_code = self.get_code(m, r)
        if diff1 <= diff2:
            return cur_code.decode(codeword1)
        return cur_code.decode(codeword2)

    def decode_first_degree(self, array):
        """
        Apply an in-place style Walsh-Hadamard transform over +/-1 representation.
        """
        array = [-1 if x == 1 else 1 if x == 0 else x for x in array]
        iterations = int(math.log(len(array), 2))
        n = len(array)

        for it in range(iterations):
            step = 2**it
            new_array = array[:]
            for i in range(0, n, 2 * step):
                for j in range(step):
                    new_array[i + j] = array[i + j] + array[i + j + step]
                    new_array[i + j + step] = array[i + j] - array[i + j + step]
            array = new_array

        return array

    def final_version_decode(
        self,
        y,
        m,
        r,
        max_depth=7,
        min_m_r_diff=2,
        current_depth=0,
    ):
        """
        Recursive split decoder with depth and (m - r) thresholds to limit recursion.
        """
        if current_depth >= max_depth or (m - r) < min_m_r_diff:
            return self.get_code(m, r).decode(y)

        n = len(y)
        assert n == 2**m, f"Word length must be 2^m={2**m}, got {n}"

        mid = n // 2
        y_l, y_r = y[:mid], y[mid:]

        s = np.bitwise_xor(y_l, y_r)

        if r - 1 > 1 and (m - r) >= min_m_r_diff:
            v_hat = self.final_version_decode(
                s,
                m - 1,
                r - 1,
                max_depth=max_depth,
                min_m_r_diff=min_m_r_diff,
                current_depth=current_depth + 1,
            )
        elif r - 1 == 1:
            v_hat = self.real_decode_first_degree(s)
        else:
            v_hat = self.get_code(m - 1, r - 1).decode(s)

        v_cw = self.get_code(m - 1, r - 1).encode(v_hat)

        y_l_corr = np.bitwise_xor(y_r, v_cw)
        y_l_orig = y_l.copy()

        if (m - r) >= min_m_r_diff:
            u_hat1 = self.final_version_decode(
                y_l_corr,
                m - 1,
                r,
                max_depth=max_depth,
                min_m_r_diff=min_m_r_diff,
                current_depth=current_depth + 1,
            )
            u_hat2 = self.final_version_decode(
                y_l_orig,
                m - 1,
                r,
                max_depth=max_depth,
                min_m_r_diff=min_m_r_diff,
                current_depth=current_depth + 1,
            )
        else:
            u_code = self.get_code(m - 1, r)
            u_hat1 = u_code.decode(y_l_corr)
            u_hat2 = u_code.decode(y_l_orig)

        u_code = self.get_code(m - 1, r)
        u1_cw = u_code.encode(u_hat1)
        cw1 = np.concatenate((u1_cw, np.bitwise_xor(u1_cw, v_cw)))

        u2_cw = u_code.encode(u_hat2)
        cw2 = np.concatenate((u2_cw, np.bitwise_xor(u2_cw, v_cw)))

        diff1 = np.count_nonzero(y != cw1)
        diff2 = np.count_nonzero(y != cw2)

        best_cw = cw1 if diff1 <= diff2 else cw2
        return self.get_code(m, r).decode(best_cw)