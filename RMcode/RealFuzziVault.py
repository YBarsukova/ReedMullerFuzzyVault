import math
from math import comb
import RMCore
import numpy as np
import hashlib
import mpmath as mp
mp.dps = 80
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
binomial_cache = {}

def cached_binomial(n, k):
    if (n, k) not in binomial_cache:
        binomial_cache[(n, k)] = mp.binomial(n, k)
    return binomial_cache[(n, k)]

def efficient_binomial(n, k):
    if (n, k) in binomial_cache:
        return binomial_cache[(n, k)]
    if k == 0 or n == k:
        binomial_cache[(n, k)] = mp.mpf(1)
        return mp.mpf(1)
    result = mp.mpf(1)
    k = min(k, n - k)
    for i in range(1, k + 1):
        result *= (n - i + 1)
        result /= i
    binomial_cache[(n, k)] = result
    return result
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

import os
import mpmath as mp

# Глобальный кэш (для словаря) — при желании можно оформить не как глобальный
_data_cache = None

def init_data(m_c, filename="results13_2.txt"):
    """
    Читает файл и формирует словарь:
      ключ = (первая_колонка + rm.code.mistakes_count)
      значение = вторая_колонка (float).
    Возвращает словарь.
    """
    data_dict = {}
    # Открываем файл
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

                # Смещаем ключ на rm.code.mistakes_count
                error_count_offset = error_count_raw + m_c
                data_dict[error_count_offset] = probability
            except ValueError:
                # Если что-то не парсится — пропускаем
                pass
    return data_dict

def get_probability(rm, num_errors):
    """
    Достаём из готового словаря значения по ключу = num_errors.
    Если ключа нет — возвращаем None или 0.0/1.0 по логике пользователя.
    """
    global _data_cache
    if _data_cache is None:
        # Инициализируем кэш 1 раз (можно делать и при старте программы)
        _data_cache = init_data(rm, "results13_2.txt")
        if not _data_cache:
            raise ValueError("Файл пуст или не содержит корректных данных.")

    # Теперь работаем с _data_cache
    # Хотите логику 'меньше min -> 1.0; больше max -> 0.0' — добавим:
    keys_sorted = sorted(_data_cache.keys())
    min_key = keys_sorted[0]
    max_key = keys_sorted[-1]

    if num_errors < min_key:
        return 1.0
    if num_errors > max_key:
        return 0.0

    # Если прямое попадание
    if num_errors in _data_cache:
        return _data_cache[num_errors]

    # Если нет точного ключа (между min_key и max_key)
    # Возвращаем None (или другой вариант, например 0.0):
    return None

def get_probability_user_version(num_errors, garant, filename="results13_2.txt"):
    data = load_data(garant, filename)
    if not data:
        raise ValueError("Файл пуст или не содержит корректных данных.")
    max_error = max(data.keys())
    min_error = min(data.keys())
    max_value=max(data.values())
    if num_errors in data:
        return data[num_errors]
    if num_errors < min_error:
        return max_value
    if num_errors > max_error:
        return 0.0
    return None
def first_filter(r, m, max_probability):
    rm = RMCore.RMCore(m, r)
    possible_set = set()
    for t in range(1, int(rm.code.n * 0.5)):
        flag = True
        for s in range(1, t):
            result = mp.mpf(0)
            bin_n_s = efficient_binomial(rm.code.n, s)
            for a in range(s + 1):
                prob_value = t + s - 2 * a
                prob = get_probability(rm.code.mistakes_count,prob_value)
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
    numerator = mp.binomial(n - t, k)
    denominator = mp.binomial(n, k)
    p = numerator / denominator
    return p

def second_filter(r, m, max_prob):
    rm = RMCore.RMCore(m, r)
    poss_set = set()
    max_prob_mpf = mp.mpf(max_prob)
    for t in range(int(rm.code.mistakes_count), int(rm.code.n * 0.5)):
        p = compute_p(rm.code.n, t, rm.code.k)
        if p <= max_prob_mpf:
            poss_set.add(t)
    return poss_set
def third_filter(r,m, max_prob):
    rm = RMCore.RMCore(m, r)
    poss_set = set()
    mc=rm.code.mistakes_count
    for t in range(rm.code.mistakes_count, int(rm.code.n*0.5)):
        if get_probability(t,mc)<=max_prob:
            poss_set.add(t)
    return poss_set
def evaluate_count_of_flipped(m,r, max_prob):
    intersection =second_filter(r, m, max_prob) & third_filter(r, m, max_prob)
    if not intersection:
        raise ValueError("Нет подходящих значений flipped: пересечение фильтров пустое.")
    print(3688 in intersection)
    return min(intersection)
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

    # def user_filter(r,m):  # TODO до 0.95 можем позволить, выбираем из полученных сетов от первых двух фильтров (не для т, а для количества возможных ошибок пользователя)
    #
    #     return 1


