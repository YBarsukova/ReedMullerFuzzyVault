import time
from datetime import datetime

from numba import typeof

import RM
import Testing
import Exel
import RMCore
from RMcode import FuzzyVault
from RMcode.FuzzyVault import Vault
from RMcode.Testing import test_splited_unlock_for_error_count, test_decode_recursed_splited, test_decode_2
import rm_code_tuning
from RMcode.rm_code_tuning import guess_mistakes
import RealFuzziVault

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
        prev = Testing.test_splited_unlock_for_error_count(vault, i)
        i += 1
        statistics.append(str(i) + " " + str(prev))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    Exel.update_excel_with_data("test_data_splited.xlsx", statistics)
def measure_main_execution_time_deterministic(c):
    start_time = time.time()
    message_length = RM.sum_binomial(c.m, c.r)
    message = generate_deterministic_message(11, message_length)
    max_value = 2 ** message_length
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
    message_length = RM.sum_binomial(core.code.m, core.code.r)
    max_value = 2 ** message_length
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
    c = RM.RM(4, 2)
    message_length = RM.sum_binomial(c.m, c.r)
    # for i in range(1,2):
    #     message = generate_deterministic_message(i, message_length)
    #     results = Testing.run_tests_for_error_counts(c, message)
    #     print(results)
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
            if (message!=d):
                count+=1
            #assert message == d, f"Decoded message does not match the original message for input {message}."
    print(count)
    print(max_value)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

def measure_main_execution_time_for_core():
    start_time = time.time()
    core=RMCore.RMCore(6,2)
    message_length = RM.sum_binomial(core.code.m, core.code.r)
    # for i in range(1,2):
    #     message = generate_deterministic_message(i, message_length)
    #     results = Testing.run_tests_for_error_counts(c, message)
    #     print(results)
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
            #assert message == d, f"Decoded message does not match the original message for input {message}."
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
        prev= Testing.tests_for_a_certain_number_of_errors_parallel2(code, i)
        #print(i,prev)
        i+=1
        statistics.append(str(i)+" "+str(prev))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    Exel.update_excel_with_data("test_data.xlsx", statistics)
def test_random2(code):
    start_time = time.time()
    prev = 1
    i=code.mistakes_count+1
    statistics = ["(" + str(code.r) + " " + str(code.m) + ")"]
    while prev > 0:
        prev= Testing.tests_for_a_certain_number_of_errors_parallel2_with_added_break(code, i)
        #print(i,prev)
        i+=1
        statistics.append(str(i)+" "+str(prev))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    Exel.update_excel_with_data("test_data.xlsx", statistics)
def test_random_rm_core(core):
    start_time = time.time()
    prev = 1
    i=core.code.mistakes_count+1
    statistics = ["(" + str(core.code.r) + " " + str(core.code.m) + ")"]
    while prev > 0:
        prev= Testing.tests_rm_core(core, i)
        #print(i,prev)
        i+=1
        statistics.append(str(i)+" "+str(prev))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    Exel.update_excel_with_data("test_recursed.xlsx", statistics)
def test_random_rm_core_starts_in_the_end(core):
    start_time = time.time()
    statistics = ["(" + str(core.code.r) + " " + str(core.code.m) + ")"]
    max_errors = core.code.n //2
    for i in range(max_errors, 0, -1):  # Тестирование с конца
        res = Testing.tests_rm_core(core, i)
        statistics.append(str(i)+" "+str(res))
        if res >= 1-10**(-3):
            for j in range(i - 1, 0, -1):
                statistics.append(f"{j} 1.0")
            break
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    Exel.update_excel_with_data("test_recursed.xlsx", statistics)
def test_txt(core):
    with open("results.txt", "a", encoding="utf-8") as file:
        file.write(str(core.code.m)+" "+str(core.code.r)+'\n')
    for i in range(core.code.n//2, 0, -1):  # Тестирование с конца
        result = Testing.tests_rm_core(core, i)
        with open("results.txt", "a", encoding="utf-8") as file:
            file.write(str(i)+" "+str(result) + "\n")
        print(f"Записан результат {i + 1}: {result}")
        if result >= 0.99:
            with open("results.txt", "a", encoding="utf-8") as file:
                for j in range(i - 1, 0, -1):
                        file.write(str(i) + " " + str(result) + "\n")
                break
    with open("results.txt", "a", encoding="utf-8") as file:
        file.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

def main():
    #measure_main_execution_time
    # code=RM.RM(6,2)
    # measure_main_execution_time_deterministic(code)
    #test_random(code)
    #code = RM.RM(localr, localm)
    # measure_main_execution_time_deterministic(code)
    # common_codes=[RM.RM(10,1),RM.RM(10,2), RM.RM(10,3)]
    # for code in common_codes:
    #     test_random2(code)
    #     print(f"Закончили вычисления вероятности для кода ({code.m}, {code.r})")
    # V=FuzzyVault.Vault(6,2)
    # vault_tests(V)
    # common_vaults=[FuzzyVault.Vault(5,2),FuzzyVault.Vault(5,3), FuzzyVault.Vault(6,2),FuzzyVault.Vault(6,3), FuzzyVault.Vault(7,2),FuzzyVault.Vault(7,3)]
    # for code in common_vaults:
    #     vault_tests(code)
    #     print(f"Закончили вычисления вероятности для кода ({code.code.m}, {code.code.r})")
    # message_length = RM.sum_binomial(4,2)
    # V.lock(generate_deterministic_message(1, message_length))
    # while True:
    #     sec=read_set_from_console()
    #     V.unlock(sec)
    # V.unlock([]
    # print(comp(4,2))
    # common_cores=[RMCore.RMCore(12,3), RMCore.RMCore(13,2), RMCore.RMCore(14,2) ]
    # for core in common_cores:
    #      test_txt(core)
    #      print(f"Закончили вычисления вероятности для кода ({core.code.m}, {core.code.r})")
    # core=RMCore.RMCore(13,3)
    # rm_code_tuning.guess_real(core, pow(2, -80))
    # print(rm.mistakes_count)
    # print(rm.get_matrix(4,1))
    #measure_main_execution_time_for_core()
    FV=RealFuzziVault.FuzziVault(4,2, 3)
    message=[1]*11
    kee=[0]
    FV.lock(message, kee)
    kee1 = [0]
    FV.unlock(kee1)

if __name__ == '__main__':
    main()