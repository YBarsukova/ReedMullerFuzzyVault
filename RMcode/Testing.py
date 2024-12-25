import itertools
import math
import multiprocessing
import random
from random import randint
import numpy as np
from numba import typeof
from tqdm import tqdm
import RM
import RMCore
from concurrent.futures import ProcessPoolExecutor

from RMcode.RM import sum_binomial


def generate_error_combinations(message_length, num_errors):
    positions = range(message_length)
    return list(itertools.combinations(positions, num_errors))
def get_random_combinations(message_length, num_errors, num_samples):
    all_combinations = generate_error_combinations(message_length, num_errors)
    return random.sample(all_combinations, min(num_samples, len(all_combinations)))
def generate_one_random_combination(message_length, num_errors):
    return tuple(random.sample(range(message_length), num_errors))
def apply_errors(encoded_message, error_positions):

    corrupted_message = encoded_message.copy()
    for ind in error_positions:
        corrupted_message[ind] ^= 1
    return corrupted_message

def test_error_correction(code, message, num_errors):
    encoded_message = code.encode(message)
    message_length = len(encoded_message)
    error_combinations = generate_error_combinations(message_length, num_errors)
    total_tests = len(error_combinations)
    successful_corrections = 0

    for error_positions in tqdm(error_combinations, desc=f"Testing {num_errors} errors"):
        corrupted_message = apply_errors(encoded_message, error_positions)
        decoded_message = code.decode(corrupted_message)
        if decoded_message == message:
            successful_corrections += 1

    unsuccessful_corrections = total_tests - successful_corrections
    return successful_corrections, unsuccessful_corrections

def run_tests_for_error_counts(code, message):
    max_errors = code.n
    message_length = len(code.encode(message))
    max_errors = int(max_errors)

    results = []

    for num_errors in range(int(code.mistakes_count), max_errors):
        total_combinations = math.comb(message_length, num_errors)
        if total_combinations > 1e6:
            print(f"Слишком много комбинаций для {num_errors} ошибок ({total_combinations} комбинаций). Пропуск...")
            continue
        successful, unsuccessful = test_error_correction(code, message, num_errors)
        results.append((num_errors, successful, unsuccessful))
        print(f"\n Для {num_errors} ошибок: Успешно исправлено {successful} из {successful + unsuccessful}")
        if successful==0:
            break

    return results
def tests_for_a_certain_number_of_errors(code, count):
    fail_count=0
    total_count=0
    while (fail_count<1000):
        total_count+=1
        message = [random.choice([0,1]) for i in range(RM.sum_binomial(code.m, code.r))]
        encoded=code.encode(message)
        emessage=apply_errors(encoded, generate_one_random_combination(code.n, count))
        if code.decode(emessage)!=message:
            fail_count+=1
    return 1-(fail_count/total_count)


def worker(code, count, fail_limit):
    fail_count = 0
    total_count = 0
    while fail_count < fail_limit:
        total_count += 1
        message = [random.choice([0, 1]) for _ in range(RM.sum_binomial(code.m, code.r))]
        encoded = code.encode(message)
        emessage = apply_errors(encoded, generate_one_random_combination(code.n, count))
        if code.decode(emessage) != message:
            fail_count += 1
    return total_count
def worker2(code, count, fail_limit, total_limit):
    fail_count = 0
    total_count = 0
    for i in range(total_limit):
        total_count += 1
        message = [random.choice([0, 1]) for _ in range(RM.sum_binomial(code.m, code.r))]
        encoded = code.encode(message)
        emessage = apply_errors(encoded, generate_one_random_combination(code.n, count))
        if code.decode(emessage) != message:
            fail_count += 1
        if fail_count>=fail_limit:
            break
    return (total_count, fail_count)
def tests_for_a_certain_number_of_errors_parallel(code, count, num_processes=4):
    fail_limit_per_process = 1000 // num_processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(worker, [(code, count, fail_limit_per_process) for _ in range(num_processes)])
    total_count = sum(results)
    return 1 - (1000/ total_count)

def tests_for_a_certain_number_of_errors_parallel2(code, count, num_processes=4):
    fail_limit_per_process = 1024 // num_processes
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(worker, code, count, fail_limit_per_process) for _ in range(num_processes)]
        results = [future.result() for future in futures]

    total_count = sum(results)
    res=1 - (1024/ total_count)
    if res<0:
        print(total_count)
    return res
def tests_for_a_certain_number_of_errors_parallel2_with_added_break(code, count, num_processes=4):
    fail_limit_per_process = 1024 // num_processes
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(worker2, code, count, fail_limit_per_process, 10**5) for _ in range(num_processes)]
        results = [future.result() for future in futures]
    total_count=0
    failed_count=0
    for elem  in results:
        total_count+=elem[0]
        failed_count+=elem[1]
    res=1 - (failed_count/ total_count)
    if res<0:
        print(total_count)
    return res
def generate_one_random_combination(message_length, num_errors):
    return tuple(random.sample(range(message_length), num_errors))
def remove_coordinates(sequence, num_coords):
    if num_coords > len(sequence):
        raise ValueError("Количество координат для удаления превышает длину последовательности.")
    indices_to_remove = random.sample(range(len(sequence)), num_coords)
    updated_sequence = [x for i, x in enumerate(sequence) if i not in indices_to_remove]
    return updated_sequence
def test_splited_unlock_for_error_count(vault, num_errors, fail_limit=1000):
    fail_count = 0
    total_count = 0
    while fail_count < fail_limit:
        total_count += 1
        message = list(map(int, bin(random.randint(0, vault.code.n))[2:].zfill(vault.code.k)))
        secret=vault.lock(message)
        if num_errors * 2 > len(secret):
            print(f"Количество ошибок {num_errors * 2} превышает длину секрета {len(secret)}.")
            return 0
        corrupted_message = remove_coordinates(secret, num_errors*2)
        success = vault.infinite_unlock(set(corrupted_message))
        if not success:
            fail_count += 1
    success_rate = 1-(fail_count/ total_count)
    print(f"Success rate for {num_errors} errors: {success_rate}")
    return success_rate


def test_decode_recursed_splited(rm_core, num_tests):
    m = rm_core.code.m
    n = 2 ** m
    for i in range(num_tests):
        message = [random.choice([0, 1]) for _ in range(sum_binomial(rm_core.code.m, rm_core.code.r))]
        encoded_message = rm_core.encode(message)
        decoded_message = rm_core.decode_recursed_splited(encoded_message)
        print(f'm: {message},\n \n {decoded_message}')

def test_decode_2(rm_core, num_tests):
    m = rm_core.code.m
    n = 2 ** m
    for i in range(num_tests):
        message = [random.choice([0, 1]) for _ in range(sum_binomial(rm_core.code.m, rm_core.code.r))]
        encoded_message = rm_core.encode(message)
        decoded_message = rm_core.final_version_decode(encoded_message, m, rm_core.code.r)
        print(f'm: {message},\n, \n {decoded_message}')
def worker3(core, count, fail_limit, total_limit):
    fail_count = 0
    total_count = 0
    c=RM.sum_binomial(core.code.m, core.code.r)
    for i in range(total_limit):
        total_count += 1
        message = [random.choice([0, 1]) for _ in range(c)]
        encoded = core.encode(message)
        emessage = apply_errors(encoded, generate_one_random_combination(core.code.n, count))
        if core.real_final_version_decode(emessage, 10) != message:
            fail_count += 1
        if fail_count>=fail_limit:
            break
    return (total_count, fail_count)
def tests_rm_core(core, count, num_processes=80):
    fail_limit_per_process = 960// num_processes
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(worker3, core, count, fail_limit_per_process, 10**5) for _ in range(num_processes)]
        results = [future.result() for future in futures]
    total_count=0
    failed_count=0
    for elem  in results:
        total_count+=elem[0]
        failed_count+=elem[1]
    res=1 - (failed_count/ total_count)
    if res<0:
        print(total_count)
    return res