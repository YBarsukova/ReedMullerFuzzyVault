import math
from math import comb
import RMCore
import numpy as np
import hashlib
def sha256_hash(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    elif isinstance(data, int):
        data = str(data).encode('utf-8')
    elif isinstance(data, list):
        data = str(data).encode('utf-8')
    elif not isinstance(data, bytes):
        raise TypeError(f"Неподдерживаемый тип данных для хеширования: {type(data)}")

    return hashlib.sha256(data).hexdigest()
def invert_bits(binary_array, indices):
    binary_array = np.array(binary_array, dtype=np.uint8)
    indices = np.array(indices, dtype=np.int32)
    valid_indices = indices[(indices >= 0) & (indices < len(binary_array))]
    binary_array[valid_indices] ^= 1
    return binary_array

_data_cache = None

def load_data(garant, filename="results13_2.txt"):
    global _data_cache
    if _data_cache is None:
        _data_cache = {}
        with open(filename, "r") as f:
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
                    _data_cache[error_count] = probability
                except ValueError:
                    continue
    return _data_cache


def get_probability(num_errors, garant, filename="results13_2.txt"):
    data = load_data(garant, filename)
    if not data:
        raise ValueError("Файл пуст или не содержит корректных данных.")
    max_error = max(data.keys())
    min_error = min(data.keys())
    if num_errors in data:
        return data[num_errors]
    if num_errors < min_error:
        return 1.0
    if num_errors > max_error:
        return 0.0
    return None
def first_filter(r,m, max_probability):
    rm=RMCore.RMCore(m,r)
    possible_set=set()
    for t in range(rm.code.mistakes_count, int(rm.code.n*0.15)): #TODO, проход по количеству ошибок
        for s in range(1, t): #TODO спросить от какой границы
            #TODO MPMATH
            # result = sum(math.comb(t, a) * comb(t, s - a) * get_probability(t + s - 2 * a, rm.code.mistakes_count) for a in range(s + 1))
            # result /= comb(rm.code.n, s)
            result = compute_probability(t,s,rm.code.mistakes_count, rm.code.n)
        if (result<=max_probability):
            possible_set.add(t)
            #TODO поставить break
    return possible_set
def log_comb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
def compute_probability(t, s, mistakes_count, n):
    terms_log = []
    for a in range(s + 1):
        p = get_probability(t + s - 2 * a, mistakes_count)
        if p <= 0:
            term_log = -math.inf
        else:
            term_log = log_comb(t, a) + log_comb(t, s - a) + math.log(p)
        terms_log.append(term_log)
    max_log = max(terms_log)
    sum_exp = sum(math.exp(l - max_log) for l in terms_log if l != -math.inf)
    log_sum = max_log + math.log(sum_exp)
    log_denom = log_comb(n, s)
    log_result = log_sum - log_denom
    result = math.exp(log_result)
    return result

def compute_p(n, t, k):
    log_numerator = log_comb(n - t, k)
    log_denominator = log_comb(n, k)
    log_p = log_numerator - log_denominator
    p = math.exp(log_p)
    return p
def second_filter(r,m, max_prob): #пытаемся угадать верные координаты
    rm=RMCore.RMCore(m,r)
    poss_set=set()
    for t in range(rm.code.mistakes_count, int(rm.code.n*0.75)):
        if (compute_p(rm.code.n, t,rm.code.k) <= max_prob):
            poss_set.add(t)
    return poss_set

def third_filter(r,m): #все хранилище закидываем
    return 1
def user_filter(r,m):#TODO до 0.95 можем позволить, выбираем из полученных сетов от первых двух фильтров (не для т, а для количества возможных ошибок пользователя)
    #чисто для себя
    return 1
def evaluate_count_of_flipped(m,r):
    intersection = first_filter(r,m) & second_filter(r,m) & user_filter(r,m)
    #return min(intersection)
    return 1
class FuzzyVault():
    def __init__(self, m, r, attempts_count):
        self.code=RMCore.RMCore(m,r)
        self.attempts_count=attempts_count
        self.is_locked=False
        self.kee_length=evaluate_count_of_flipped(m,r)
        print("Your vault has been created \n Kee length must be "+str(self.kee_length)+"\n")
    def lock(self, secret, kee):
        if self.is_locked:
            print("Already locked \n")
        else:
            if(len(kee)!=self.kee_length):
                print("Wrong kee length, try again \n")
                exit()
            self.hash=sha256_hash(secret)
            self.vault=invert_bits(self.code.encode(secret), kee)
            self.is_locked=True
            print("Vault has been locked \n")
    def unlock(self, kee):
        cur_vault=invert_bits(self.vault,kee)
        decoded = self.code.real_final_version_decode(cur_vault)
        if(self.hash==sha256_hash(decoded)):
            print("Vault has been unlocked!!! \n ")
        else:
            self.attempts_count-=1
            print("Vault hasn't been unlocked, now u have only " + str(self.attempts_count)+ " attempts \n")



