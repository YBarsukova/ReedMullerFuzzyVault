import time
import RM
import Testing
from RMcode.Testing import tests_for_a_certain_number_of_errors


def generate_deterministic_message(number, length):
    binary_message = list(map(int, bin(number)[2:].zfill(length)))
    return binary_message


def measure_main_execution_time_deterministic(c):
    start_time = time.time()
    message_length = RM.sum_binomial(c.m, c.r)
    message = generate_deterministic_message(42, message_length)
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
    prev = 1
    i=code.mistakes_count+1
    while prev > 0:
        prev=tests_for_a_certain_number_of_errors(code, i)
        print(i,prev)
        i+=1

def main():
    #measure_main_execution_time
    code=RM.RM(10,2)
    measure_main_execution_time_deterministic(code)
    #test_random(code)
if __name__ == '__main__':
    main()