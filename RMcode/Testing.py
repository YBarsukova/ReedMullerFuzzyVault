import itertools
import math

import numpy as np
from tqdm import tqdm
import RM

def generate_error_combinations(message_length, num_errors):

    positions = range(message_length)
    return list(itertools.combinations(positions, num_errors))

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
        if (successful==0):
            break

    return results
