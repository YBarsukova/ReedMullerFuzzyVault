import time

from datetime import datetime

from fuzzy_vault.fuzzy_vault import Vault
from fuzzy_vault.real_fuzzy_vault import FuzzyVault
from rm_code.combined_rm import CombinedRM
from rm_code.reedmuller import ReedMuller
from rm_code.rm_code_tuning import guess_real
from rm_core import RMCore
from tests import testing
from tests.testing import *

from util.excel_module import update_excel_with_data

def print_map(map_data):
    for key, value in map_data.items():
        key_str = ''.join(map(str, key))
        value_str = ''.join(map(str, value))
        print(f"Key: {key_str}, Value: {value_str}")

def generate_deterministic_message(number, length):
    binary_message = list(map(int, bin(number)[2:].zfill(length)))
    return binary_message
def read_set_from_console():
    input_data = input("Введите элементы множества, разделённые запятой и пробелом: ")
    number_set = set(map(int, input_data.split(', ')))
    return number_set
def vault_tests(vault):
    start_time = time.time()
    prev = 1
    i = 1
    statistics = ["(" + str(vault.code.r) + " " + str(vault.code.m) + ")"]
    while prev > 0:
        prev = test_splited_unlock_for_error_count(vault, i)
        i += 1
        statistics.append(str(i) + " " + str(prev))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    update_excel_with_data("../out/test_data_splited.xlsx", statistics)
def measure_main_execution_time_deterministic(c):
    start_time = time.time()
    message_length = sum_binomial(c.m, c.r)
    for repeat in range(1):
        for number in range(100):
            message = generate_deterministic_message(number, message_length)
            emessage = c.encode(message)
            emessage[2] = emessage[2] ^ 1
            for j in range(3):
                emessage[j] = 3
            d = c.decode(emessage)
            assert message == d, f"Decoded message does not match the original message for input {message}."
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

def measure_main_execution_time_deterministic_for_rmcore(core):
    start_time = time.time()
    message_length = sum_binomial(core.code.m, core.code.r)
    for repeat in range(100):
        for number in range(26, 100):
            message = generate_deterministic_message(number, message_length)
            emessage = core.code.encode(message)
            emessage[2] = emessage[2] ^ 1
            for j in range(3):
                emessage[j] = 3
            d = core.real_final_version_decode(emessage )
            assert message == d, f"Decoded message does not match the original message for input {message, d}."
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

def measure_main_execution_time():
    start_time = time.time()
    c = ReedMuller(4, 2)
    message_length = sum_binomial(c.m, c.r)
    max_value = 2 ** message_length
    count=0
    for repeat in range(1):
        for number in range(max_value):
            message = generate_deterministic_message(number, message_length)
            emessage = c.encode(message)
            emessage[2] = emessage[2] ^ 1
            for j in range(0,2):
                emessage[j] = 3
            d = c.decode(emessage)
            if message!=d:
                count+=1
    print(count)
    print(max_value)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

def measure_main_execution_time_for_core():
    start_time = time.time()
    core=RMCore(6,2)
    message_length = sum_binomial(core.code.m, core.code.r)
    max_value =1000
    count=0
    for repeat in range(1):
        for number in range(max_value):
            message = generate_deterministic_message(number, message_length)
            emessage = core.code.encode(message)
            emessage[2] = emessage[2] ^ 1
            for j in range(0,2):
                emessage[j] = 3
            d = core.real_final_version_decode(emessage, 3)
            if (message!=d):
                count+=1
    print(count)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
def test_random(code):
    start_time = time.time()
    prev = 1
    i=code.mistakes_count+1
    statistics = ["(" + str(code.r) + " " + str(code.m) + ")"]
    while prev > 0:
        prev= tests_for_a_certain_number_of_errors_parallel2(code, i)
        i+=1
        statistics.append(str(i)+" "+str(prev))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    update_excel_with_data("../out/test_data.xlsx", statistics)
def test_random2(code):
    start_time = time.time()
    prev = 1
    i=code.mistakes_count+1
    statistics = ["(" + str(code.r) + " " + str(code.m) + ")"]
    while prev > 0:
        prev= tests_for_a_certain_number_of_errors_parallel2_with_added_break(code, i)
        i+=1
        statistics.append(str(i)+" "+str(prev))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    update_excel_with_data("../out/test_data.xlsx", statistics)
def test_random_rm_core(core):
    start_time = time.time()
    prev = 1
    i=core.code.mistakes_count+1
    statistics = ["(" + str(core.code.r) + " " + str(core.code.m) + ")"]
    while prev > 0:
        prev= tests_rm_core(core, i)
        i+=1
        statistics.append(str(i)+" "+str(prev))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    update_excel_with_data("../out/test_recursed.xlsx", statistics)
def test_random_rm_core_starts_in_the_end(core):
    start_time = time.time()
    statistics = ["(" + str(core.code.r) + " " + str(core.code.m) + ")"]
    max_errors = core.code.n //2
    for i in range(max_errors, 0, -1):  # Тестирование с конца
        res = tests_rm_core(core, i)
        statistics.append(str(i)+" "+str(res))
        if res >= 1-10**(-3):
            for j in range(i - 1, 0, -1):
                statistics.append(f"{j} 1.0")
            break
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    update_excel_with_data("../out/test_recursed.xlsx", statistics)
def test_txt(core):
    with open("../out/results.txt", "a", encoding="utf-8") as file:
        file.write(str(core.code.m)+" "+str(core.code.r)+'\n')
    for i in range(core.code.n//2, 0, -1):  # Тестирование с конца
        result = tests_rm_core(core, i)
        with open("../out/results.txt", "a", encoding="utf-8") as file:
            file.write(str(i)+" "+str(result) + "\n")
        print(f"Записан результат {i + 1}: {result}")
        if result >= 0.99:
            with open("../out/results.txt", "a", encoding="utf-8") as file:
                for j in range(i - 1, 0, -1):
                        file.write(str(i) + " " + str(result) + "\n")
                break
    with open("../out/results.txt", "a", encoding="utf-8") as file:
        file.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
def test_for_combined(core):
    with open("../out/results_for_majory.txt", "a", encoding="utf-8") as file:
        file.write(str(core.m)+" "+str(core.r)+'\n')
    for i in range(core.n//2, 0, -10):
        print("start with "+ str(i)+"\n")
        result = test_combined(core, i)
        with open("../out/results_for_majory.txt", "a", encoding="utf-8") as file:
            file.write(str(i)+" "+str(result) + "\n")
        print(f"Записан результат {i + 1}: {result}")
        if result >= 0.99:
            with open("../out/results_for_majory.txt", "a", encoding="utf-8") as file:
                for j in range(i - 1, 0, -1):
                        file.write(str(i) + " " + str(result) + "\n")
                break
    with open("../out/results_for_majory.txt", "a", encoding="utf-8") as file:
        file.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

def main():
    #measure_main_execution_time
    code=ReedMuller(6,2)
    measure_main_execution_time_deterministic(code)
    test_random(code)
    #code = ReedMuller(localr, localm)
    measure_main_execution_time_deterministic(code)
    common_codes=[ReedMuller(10,1),ReedMuller(10,2), ReedMuller(10,3)]
    for code in common_codes:
        test_random2(code)
        print(f"Закончили вычисления вероятности для кода ({code.m}, {code.r})")
    v=Vault(6,2)
    vault_tests(v)
    common_vaults=[Vault(5,2), Vault(5,3), Vault(6,2),Vault(6,3), Vault(7,2),Vault(7,3)]
    for code in common_vaults:
        vault_tests(code)
        print(f"Закончили вычисления вероятности для кода ({code.code.m}, {code.code.r})")
    message_length = sum_binomial(4,2)
    v.lock(generate_deterministic_message(1, message_length))
    # while True:
    #     sec=read_set_from_console()
    #     v.unlock(sec)
    v.unlock([])
    common_cores=[RMCore(12,3), RMCore(13,2), RMCore(14,2) ]
    for core in common_cores:
         test_txt(core)
         print(f"Закончили вычисления вероятности для кода ({core.code.m}, {core.code.r})")
    core=RMCore(13,3)
    print(real_fuzzy_vault.first_filter(2,13, math.pow(2,-10)))
    print(real_fuzzy_vault.evaluate_count_of_flipped(13,2,math.pow(2, -80)))
    testing.run_all_filters(2,13,math.pow(2,-80))
    core=CombinedRM(4,2)
    test_for_combined(core)
    rm=RMCore(13,2)
    print(rm.code.mistakes_count)
    with open("../out/probs.txt",'a') as f:
        for i in range(0,int(math.pow(2,13))):
            f.write(f'{i} {real_fuzzy_vault.get_probability(i, rm.code.mistakes_count)} \n')

    core=RMCore(3,2)
    print(core.code.k)
    print(core.code.get_matrix(3,2))
    print(core.code.decode([1,0,0,1,1,1,1,0]))
    guess_real(core, pow(2, -80))
    measure_main_execution_time_for_core()
    fv=FuzzyVault(4,2, 3)
    message=[1]*11
    kee=[0]
    fv.lock(message, kee)
    kee1 = [0]
    fv.unlock(kee1)

if __name__ == '__main__':
    main()