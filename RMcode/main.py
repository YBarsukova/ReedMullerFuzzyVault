import time
import RM
import Testing

def generate_deterministic_message(number, length):
    binary_message = list(map(int, bin(number)[2:].zfill(length)))
    return binary_message


def measure_main_execution_time():
    start_time = time.time()
    c = RM.RM(8, 3)
    message_length = RM.sum_binomial(c.m, c.r)
    message = generate_deterministic_message(42, message_length)
    max_value = 2 ** message_length

    # for repeat in range(20):
    #
    #     for number in range(max_value):
    #         message = generate_deterministic_message(number, message_length)
    #         emessage = c.encode(message)
    #         emessage[2] = emessage[2] ^ 1
    #         for j in range(3):
    #             emessage[j] = 3
    #
    #         d = c.Decode(emessage)
    #         assert message == d, f"Decoded message does not match the original message for input {message}."
    results = Testing.run_tests_for_error_counts(c, message)
    print(results)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")


measure_main_execution_time()