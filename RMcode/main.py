import time
import RM
import Testing
import Exel
#from RMcode import FuzzyVault
def generate_deterministic_message(number, length):
    binary_message = list(map(int, bin(number)[2:].zfill(length)))
    return binary_message
def read_set_from_console():
    input_data = input("Введите элементы множества, разделённые запятой и пробелом: ")
    number_set = set(map(int, input_data.split(', ')))
    return number_set

def measure_main_execution_time_deterministic(c):
    start_time = time.time()
    message_length = RM.sum_binomial(c.m, c.r)
    message = generate_deterministic_message(42, message_length)
    max_value = 2 ** message_length
    for repeat in range(100):
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
def main():
    #measure_main_execution_time
    # code=RM.RM(6,2)
    # measure_main_execution_time_deterministic(code)
    #test_random(code)
    #code = RM.RM(localr, localm)
    # measure_main_execution_time_deterministic(code)
    common_codes=[RM.RM(10,1),RM.RM(10,2), RM.RM(10,3)]
    for code in common_codes:
        test_random2(code)
        print(f"Закончили вычисления вероятности для кода ({code.m}, {code.r})")
    # V=FuzzyVault.Vault(4,1)
    # message_length = RM.sum_binomial(4,1)
    # V.lock(generate_deterministic_message(1, message_length))
    # while True:
    #     sec=read_set_from_console()
    #     V.unlock(sec)
    # V.unlock([])
if __name__ == '__main__':
    main()