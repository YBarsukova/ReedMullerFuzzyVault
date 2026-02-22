import hashlib
import math

import mpmath as mp
import numpy as np

from rm_code.rm_core import RMCore

# Increase mpmath precision (needed for binomials/probability computations).
mp.dps = 80

# Global cache
_data_cache = None

# Global cache for binomial coefficients (mp.mpf) to avoid recomputation.
binomial_cache = {}


def sha256_hash(data):
    """
    Compute SHA-256 hash of input data (str/int/list/bytes).
    Returns a hex string.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    elif isinstance(data, int):
        data = str(data).encode("utf-8")
    elif isinstance(data, list):
        data = str(data).encode("utf-8")
    elif not isinstance(data, bytes):
        raise TypeError(f"Unsupported type for hashing: {type(data)}")

    return hashlib.sha256(data).hexdigest()


def invert_bits(binary_array, indices):
    """
    Flip bits (XOR with 1) in binary_array at positions specified by indices.
    Out-of-range indices are ignored.
    """
    binary_array = np.array(binary_array, dtype=np.uint8)
    indices = np.array(indices, dtype=np.int32)
    valid_indices = indices[(indices >= 0) & (indices < len(binary_array))]
    binary_array[valid_indices] ^= 1
    return binary_array


def cached_binomial(n, k):
    """
    Return binomial coefficient C(n, k) using the shared dictionary cache.
    """
    if (n, k) not in binomial_cache:
        binomial_cache[(n, k)] = mp.binomial(n, k)
    return binomial_cache[(n, k)]


def efficient_binomial(n, k):
    """
    Compute C(n, k) efficiently and store it in binomial_cache.
    Uses multiplicative formula to avoid calling mp.binomial too often.
    """
    if (n, k) in binomial_cache:
        return binomial_cache[(n, k)]

    if k == 0 or n == k:
        binomial_cache[(n, k)] = mp.mpf(1)
        return mp.mpf(1)

    result = mp.mpf(1)
    k = min(k, n - k)
    for i in range(1, k + 1):
        result *= n - i + 1
        result /= i

    binomial_cache[(n, k)] = result
    return result


def load_data(garant, filename="../out/results13_2.txt"):
    """
    Lazily load the probability table from a file and store it in _data_cache.

    Expected file format: "raw_error probability" per line.
    Keys in the resulting dictionary are shifted by garant:
      key = raw_error + garant
      value = probability (float)
    """
    global _data_cache
    if _data_cache is None:
        _data_cache = {}
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 2:
                    continue

                try:
                    error_count = int(parts[0]) + garant
                    probability = float(parts[1])
                except ValueError:
                    continue

                _data_cache[error_count] = probability

    return _data_cache


def init_data(mistakes_count, filename="../out/results13_2.txt"):
    """
    Read the file and build a dictionary:
      key = raw_error + mistakes_count
      value = probability (float)
    Returns a fresh dictionary (does not use _data_cache).
    """
    data_dict = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                error_count_raw = int(parts[0])
                probability = float(parts[1])
            except ValueError:
                continue

            error_count_offset = error_count_raw + mistakes_count
            data_dict[error_count_offset] = probability

    return data_dict


def _extract_mistakes_count(value):
    """
    Helper: extract mistakes_count from:
    - int/np.integer
    - an RMCore-like object with value.code.mistakes_count
    - anything that can be cast to int
    """
    if isinstance(value, (int, np.integer)):
        return int(value)
    if hasattr(value, "code") and hasattr(value.code, "mistakes_count"):
        return int(value.code.mistakes_count)
    return int(value)


def get_probability(mistakes_count_or_rm, num_errors, filename="../out/results13_2.txt"):
    """
    Get success probability for a given number of errors (num_errors).

    mistakes_count_or_rm can be:
    - mistakes_count (int)
    - an RMCore-like object from which .code.mistakes_count can be read

    Boundary behavior:
    - if num_errors < min key in the table -> 1.0
    - if num_errors > max key in the table -> 0.0
    - if inside range but missing exact key -> None
    """
    global _data_cache

    mistakes_count = _extract_mistakes_count(mistakes_count_or_rm)

    if _data_cache is None:
        _data_cache = init_data(mistakes_count, filename)
        if not _data_cache:
            raise ValueError("The file is empty or contains no valid data.")

    keys_sorted = sorted(_data_cache.keys())
    min_key = keys_sorted[0]
    max_key = keys_sorted[-1]

    if num_errors < min_key:
        return 1.0
    if num_errors > max_key:
        return 0.0

    if num_errors in _data_cache:
        return _data_cache[num_errors]

    return None


def get_probability_user_version(num_errors, garant, filename="../out/results13_2.txt"):
    """
    A version of probability lookup close to the original user logic:
    - keys are shifted by garant
    - if num_errors < min -> returns max(probabilities)
    - if num_errors > max -> 0.0
    - if missing key inside range -> None
    """
    data = load_data(garant, filename)
    if not data:
        raise ValueError("The file is empty or contains no valid data.")

    max_error = max(data.keys())
    min_error = min(data.keys())
    max_value = max(data.values())

    if num_errors in data:
        return data[num_errors]
    if num_errors < min_error:
        return max_value
    if num_errors > max_error:
        return 0.0

    return None


def first_filter(r, m, max_probability):
    """
    First filter for parameter t:
    iterate t (1..n/2) and verify for all s < t that the estimate does not exceed max_probability.

    Uses get_probability(...) and binomial coefficients.
    """
    rm = RMCore(m, r)
    possible_set = set()

    for t in range(1, int(rm.code.n * 0.5)):
        flag = True
        for s in range(1, t):
            result = mp.mpf(0)
            bin_n_s = efficient_binomial(rm.code.n, s)

            for a in range(s + 1):
                prob_value = t + s - 2 * a
                prob = get_probability(rm.code.mistakes_count, prob_value)
                if prob is None or prob == 0.0:
                    continue

                comb_product = efficient_binomial(t, a) * efficient_binomial(t, s - a)
                result += comb_product * prob

            result /= bin_n_s
            if result > max_probability:
                flag = False
                break

        if flag:
            possible_set.add(t)

    return possible_set


def log_comb(n, k):
    """
    Logarithm of the binomial coefficient ln(C(n, k)) via lgamma.
    More numerically stable than factorial for large n.
    """
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def compute_probability(t, s, mistakes_count, n):
    """
    Compute probability via log-sum-exp:
    sum_{a=0..s} C(t,a)*C(t,s-a)*p(t+s-2a) / C(n,s)

    mistakes_count is used as the shift for probability table keys.
    """
    terms_log = []
    for a in range(s + 1):
        p = get_probability(mistakes_count, t + s - 2 * a)
        if p is None or p <= 0:
            term_log = -math.inf
        else:
            term_log = log_comb(t, a) + log_comb(t, s - a) + math.log(p)
        terms_log.append(term_log)

    max_log = max(terms_log)
    sum_exp = sum(math.exp(l - max_log) for l in terms_log if l != -math.inf)
    log_sum = max_log + math.log(sum_exp)

    log_denom = log_comb(n, s)
    log_result = log_sum - log_denom
    return math.exp(log_result)


def compute_p(n, t, k):
    """
    Compute p = C(n - t, k) / C(n, k) (mpmath),
    used in the second filter.
    """
    numerator = mp.binomial(n - t, k)
    denominator = mp.binomial(n, k)
    return numerator / denominator


def second_filter(r, m, max_prob):
    """
    Second filter for parameter t:
    keep those t where C(n-t, k)/C(n,k) <= max_prob.
    """
    rm = RMCore(m, r)
    poss_set = set()
    max_prob_mpf = mp.mpf(max_prob)

    for t in range(int(rm.code.mistakes_count), int(rm.code.n * 0.5)):
        p = compute_p(rm.code.n, t, rm.code.k)
        if p <= max_prob_mpf:
            poss_set.add(t)

    return poss_set


def third_filter(r, m, max_prob):
    """
    Third filter for parameter t:
    keep those t where empirical probability get_probability(mistakes_count, t) <= max_prob.
    """
    rm = RMCore(m, r)
    poss_set = set()
    mc = rm.code.mistakes_count

    for t in range(rm.code.mistakes_count, int(rm.code.n * 0.5)):
        if get_probability(mc, t) <= max_prob:
            poss_set.add(t)

    return poss_set


def evaluate_count_of_flipped(m, r, max_prob=0.001):
    """
    Select the minimal suitable t (key length / number of flipped bits),
    as the minimum element of the intersection between second_filter and third_filter.

    max_prob is set by default (0.001) so the code works even if the argument is omitted.
    """
    intersection = second_filter(r, m, max_prob) & third_filter(r, m, max_prob)
    if not intersection:
        raise ValueError("No suitable flipped values: the intersection of filters is empty.")
    print(3688 in intersection)
    return min(intersection)


class FuzzyVault:
    """
    A fuzzy vault implementation on top of RMCore:
    - lock: encodes the secret, flips bits using the key, stores hash(secret)
    - unlock: attempts to recover the secret and verifies using SHA-256
    """

    def __init__(self, m, r, attempts_count):
        """
        Create a vault for RM(m, r) with a limited number of unlock attempts.
        key_length is computed via evaluate_count_of_flipped.
        """
        self.vault = None
        self.hash = None
        self.code = RMCore(m, r)
        self.attempts_count = attempts_count
        self.is_locked = False
        self.key_length = evaluate_count_of_flipped(m, r)
        print(f"Vault created.\nKey length must be {self.key_length}.\n")

    def lock(self, secret, key):
        """
        Lock the vault:
        - ensure the vault is not already locked
        - validate key length
        - store SHA-256(secret)
        - store vault = invert_bits(encode(secret), key)
        """
        if self.is_locked:
            print("Already locked.\n")
            return

        if len(key) != self.key_length:
            print("Wrong key length, try again.\n")
            exit()

        self.hash = sha256_hash(secret)
        self.vault = invert_bits(self.code.encode(secret), key)
        self.is_locked = True
        print("Vault has been locked.\n")

    def unlock(self, key):
        """
        Attempt to unlock the vault:
        - invert bits back using key
        - decode
        - verify hash
        - decrement attempts_count on failure
        """
        cur_vault = invert_bits(self.vault, key)
        decoded = self.code.real_final_version_decode(cur_vault)
        if self.hash == sha256_hash(decoded):
            print("Vault has been unlocked!!!\n")
        else:
            self.attempts_count -= 1
            print(
                "Vault hasn't been unlocked, now you have only "
                + str(self.attempts_count)
                + " attempts.\n"
            )