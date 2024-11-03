import itertools
import math
import random

import numpy as np
from tqdm import tqdm
import RM

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
    while (fail_count<100):
        total_count+=1
        message = [random.choice([0,1]) for i in range(RM.sum_binomial(code.m, code.r))]
        encoded=code.encode(message)
        emessage=apply_errors(encoded, generate_one_random_combination(code.n, count))
        if code.decode(emessage)!=message:
            fail_count+=1
    return 1-(fail_count/total_count)
