import time
from datetime import datetime

from rm_code.combined_rm import CombinedRM
from rm_code.reedmuller import ReedMuller
from rm_code.rm_core import RMCore
from tests.testing import (
    test_combined,
    tests_for_a_certain_number_of_errors_parallel2,
    tests_for_a_certain_number_of_errors_parallel2_with_added_break,
    tests_rm_core,
)
from util.excel_module import update_excel_with_data


def read_int(prompt, min_value=None, max_value=None, allow_empty=False, default_value=None):
    """
    Read an integer from console with validation and optional bounds.
    """
    while True:
        raw_value = input(prompt).strip()

        if raw_value == "":
            if allow_empty:
                return default_value
            print("Error: please enter an integer.")
            continue

        try:
            value = int(raw_value)
        except ValueError:
            print("Error: please enter an integer.")
            continue

        if min_value is not None and value < min_value:
            print(f"Error: value must be >= {min_value}.")
            continue

        if max_value is not None and value > max_value:
            print(f"Error: value must be <= {max_value}.")
            continue

        return value


def read_code_params():
    """
    Ask the user for Reed-Muller code parameters (m, r) and validate them.
    """
    print("\nEnter Reed-Muller code parameters RM(m, r):")
    m = read_int("m (>= 1): ", min_value=1)
    r = read_int(f"r (0..{m}): ", min_value=0, max_value=m)
    return m, r


def read_num_processes(default_value):
    """
    Ask the user for the number of parallel processes.
    """
    return read_int(
        f"Number of parallel processes [Enter = {default_value}]: ",
        min_value=1,
        allow_empty=True,
        default_value=default_value,
    )


def read_total_fail_limit(num_processes, default_value):
    """
    Ask the user for the total fail limit and ensure it is divisible by num_processes.
    """
    while True:
        total_fail_limit = read_int(
            f"Total fail limit [Enter = {default_value}]: ",
            min_value=1,
            allow_empty=True,
            default_value=default_value,
        )

        if total_fail_limit % num_processes != 0:
            print(
                "Error: total fail limit must be divisible by the number of processes "
                f"({num_processes}) without remainder."
            )
            continue

        return total_fail_limit


def read_total_limit_per_process(default_value):
    """
    Ask the user for the per-process trial limit.
    """
    return read_int(
        f"Per-process trial limit (total_limit_per_process) [Enter = {default_value}]: ",
        min_value=1,
        allow_empty=True,
        default_value=default_value,
    )


def read_decode_depth(default_value):
    """
    Ask the user for RMCore decode depth.
    """
    return read_int(
        f"RMCore decode depth (decode_depth) [Enter = {default_value}]: ",
        min_value=1,
        allow_empty=True,
        default_value=default_value,
    )


def print_menu():
    """
    Print the list of available functions with short English explanations.
    """
    print("\nChoose a function to run:")
    print(
        "1) test_random(code) [ReedMuller]\n"
        "   Runs parallel Monte Carlo tests for ReedMuller and saves stats to Excel "
        "(test_data.xlsx)."
    )
    print(
        "2) test_random2(code) [ReedMuller, early break]\n"
        "   Same as test_random, but uses an early-break variant with a per-process "
        "trial cap (usually faster)."
    )
    print(
        "3) test_random_rm_core_starts_in_the_end(core) [RMCore]\n"
        "   Tests RMCore starting from large error counts (n//2 down to 1), then fills "
        "lower counts with 1.0 after reaching a high success threshold."
    )
    print(
        "4) test_random_rm_core(core) [RMCore]\n"
        "   Runs parallel Monte Carlo tests for RMCore from mistakes_count+1 upward until "
        "the success rate reaches zero."
    )
    print(
        "5) test_txt(core) [RMCore -> out/results.txt]\n"
        "   Runs RMCore tests and appends results to a text file, then writes a timestamp "
        "at the end."
    )
    print(
        "6) test_for_combined(core) [CombinedRM -> out/results_for_majory.txt]\n"
        "   Runs CombinedRM tests (step -10 by error count), writes results to a text file, "
        "then writes a timestamp."
    )


def build_target(choice, m, r):
    """
    Build the target object required by the selected function.
    """
    if choice in (1, 2):
        return ReedMuller(m, r)
    if choice in (3, 4, 5):
        return RMCore(m, r)
    if choice == 6:
        return CombinedRM(m, r)
    raise ValueError("Unsupported menu choice.")


def read_execution_params(choice):
    """
    Read execution parameters relevant to the selected function.
    """
    if choice == 1:
        default_num_processes = 4
        default_total_fail_limit = 1024

        num_processes = read_num_processes(default_num_processes)
        total_fail_limit = read_total_fail_limit(num_processes, default_total_fail_limit)

        return {
            "num_processes": num_processes,
            "total_fail_limit": total_fail_limit,
        }

    if choice == 2:
        default_num_processes = 4
        default_total_fail_limit = 1024
        default_total_limit_per_process = 10**5

        num_processes = read_num_processes(default_num_processes)
        total_fail_limit = read_total_fail_limit(num_processes, default_total_fail_limit)
        total_limit_per_process = read_total_limit_per_process(default_total_limit_per_process)

        return {
            "num_processes": num_processes,
            "total_fail_limit": total_fail_limit,
            "total_limit_per_process": total_limit_per_process,
        }

    if choice in (3, 4, 5):
        default_num_processes = 60
        default_total_fail_limit = 1020
        default_total_limit_per_process = 10**5
        default_decode_depth = 3

        num_processes = read_num_processes(default_num_processes)
        total_fail_limit = read_total_fail_limit(num_processes, default_total_fail_limit)
        total_limit_per_process = read_total_limit_per_process(default_total_limit_per_process)
        decode_depth = read_decode_depth(default_decode_depth)

        return {
            "num_processes": num_processes,
            "total_fail_limit": total_fail_limit,
            "total_limit_per_process": total_limit_per_process,
            "decode_depth": decode_depth,
        }

    if choice == 6:
        default_num_processes = 60
        default_total_fail_limit = 1020
        default_total_limit_per_process = 10**5

        num_processes = read_num_processes(default_num_processes)
        total_fail_limit = read_total_fail_limit(num_processes, default_total_fail_limit)
        total_limit_per_process = read_total_limit_per_process(default_total_limit_per_process)

        return {
            "num_processes": num_processes,
            "total_fail_limit": total_fail_limit,
            "total_limit_per_process": total_limit_per_process,
        }

    raise ValueError("Unsupported menu choice.")


def test_random(code, num_processes=4, total_fail_limit=1024):
    """
    Run random parallel tests for ReedMuller code using tests_for_a_certain_number_of_errors_parallel2.
    Collect statistics until success rate becomes zero and save results to Excel.
    """
    start_time = time.time()
    prev = 1
    error_count = code.mistakes_count + 1
    statistics = [f"({code.r} {code.m})"]

    while prev > 0:
        prev = tests_for_a_certain_number_of_errors_parallel2(
            code,
            error_count,
            num_processes=num_processes,
            total_fail_limit=total_fail_limit,
        )
        error_count += 1
        statistics.append(f"{error_count} {prev}")

    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

    update_excel_with_data("out/test_data.xlsx", statistics)


def test_random2(
    code,
    num_processes=4,
    total_fail_limit=1024,
    total_limit_per_process=10**5,
):
    """
    Run random parallel tests for ReedMuller code using the early-break variant.
    Collect statistics until success rate becomes zero and save results to Excel.
    """
    start_time = time.time()
    prev = 1
    error_count = code.mistakes_count + 1
    statistics = [f"({code.r} {code.m})"]

    while prev > 0:
        prev = tests_for_a_certain_number_of_errors_parallel2_with_added_break(
            code,
            error_count,
            num_processes=num_processes,
            total_fail_limit=total_fail_limit,
            total_limit_per_process=total_limit_per_process,
        )
        error_count += 1
        statistics.append(f"{error_count} {prev}")

    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

    update_excel_with_data("out/test_data.xlsx", statistics)


def test_random_rm_core(
    core,
    num_processes=60,
    total_fail_limit=1020,
    total_limit_per_process=10**5,
    decode_depth=3,
):
    """
    Run random parallel tests for RMCore decoder.
    Collect statistics until success rate becomes zero and save results to Excel.
    """
    start_time = time.time()
    prev = 1
    error_count = core.code.mistakes_count + 1
    statistics = [f"({core.code.r} {core.code.m})"]

    while prev > 0:
        prev = tests_rm_core(
            core,
            error_count,
            num_processes=num_processes,
            total_fail_limit=total_fail_limit,
            total_limit_per_process=total_limit_per_process,
            decode_depth=decode_depth,
        )
        error_count += 1
        statistics.append(f"{error_count} {prev}")

    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

    update_excel_with_data("out/test_recursed.xlsx", statistics)


def test_random_rm_core_starts_in_the_end(
    core,
    num_processes=60,
    total_fail_limit=1020,
    total_limit_per_process=10**5,
    decode_depth=3,
):
    """
    Run RMCore random tests starting from the largest error count (n//2) down to 1.
    If a high success threshold is reached, fill the remaining lower error counts with 1.0.
    Save results to Excel.
    """
    start_time = time.time()
    statistics = [f"({core.code.r} {core.code.m})"]
    max_errors = core.code.n // 2

    for error_count in range(max_errors, 0, -1):
        result = tests_rm_core(
            core,
            error_count,
            num_processes=num_processes,
            total_fail_limit=total_fail_limit,
            total_limit_per_process=total_limit_per_process,
            decode_depth=decode_depth,
        )
        statistics.append(f"{error_count} {result}")

        if result >= 1 - 10 ** (-3):
            for j in range(error_count - 1, 0, -1):
                statistics.append(f"{j} 1.0")
            break

    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.4f} seconds")

    update_excel_with_data("out/test_recursed.xlsx", statistics)


def test_txt(
    core,
    num_processes=60,
    total_fail_limit=1020,
    total_limit_per_process=10**5,
    decode_depth=3,
):
    """
    Run RMCore tests for error counts from n//2 down to 1 and append results to out/results.txt.
    Writes a timestamp at the end of the file.
    """
    with open("out/results.txt", "a", encoding="utf-8") as file:
        file.write(str(core.code.m) + " " + str(core.code.r) + "\n")

    for error_count in range(core.code.n // 2, 0, -1):
        result = tests_rm_core(
            core,
            error_count,
            num_processes=num_processes,
            total_fail_limit=total_fail_limit,
            total_limit_per_process=total_limit_per_process,
            decode_depth=decode_depth,
        )

        with open("out/results.txt", "a", encoding="utf-8") as file:
            file.write(str(error_count) + " " + str(result) + "\n")

        print(f"Recorded result {error_count + 1}: {result}")

        if result >= 0.99:
            with open("out/results.txt", "a", encoding="utf-8") as file:
                for j in range(error_count - 1, 0, -1):
                    file.write(str(error_count) + " " + str(result) + "\n")
                break

    with open("out/results.txt", "a", encoding="utf-8") as file:
        file.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")


def test_for_combined(
    core,
    num_processes=60,
    total_fail_limit=1020,
    total_limit_per_process=10**5,
):
    """
    Run CombinedRM tests for error counts from n//2 down to 1 with step -10 and append
    results to out/results_for_majory.txt. Writes a timestamp at the end of the file.
    """
    with open("out/results_for_majory.txt", "a", encoding="utf-8") as file:
        file.write(str(core.m) + " " + str(core.r) + "\n")

    for error_count in range(core.n // 2, 0, -10):
        print("Starting with " + str(error_count) + "\n")

        result = test_combined(
            core,
            error_count,
            num_processes=num_processes,
            total_fail_limit=total_fail_limit,
            total_limit_per_process=total_limit_per_process,
        )

        with open("out/results_for_majory.txt", "a", encoding="utf-8") as file:
            file.write(str(error_count) + " " + str(result) + "\n")

        print(f"Recorded result {error_count + 1}: {result}")

        if result >= 0.99:
            with open("out/results_for_majory.txt", "a", encoding="utf-8") as file:
                for j in range(error_count - 1, 0, -1):
                    file.write(str(error_count) + " " + str(result) + "\n")
                break

    with open("out/results_for_majory.txt", "a", encoding="utf-8") as file:
        file.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")


def run_selected_function(choice, target, exec_params):
    """
    Dispatch to the selected function using prepared target object and execution parameters.
    """
    if choice == 1:
        test_random(
            target,
            num_processes=exec_params["num_processes"],
            total_fail_limit=exec_params["total_fail_limit"],
        )
        return

    if choice == 2:
        test_random2(
            target,
            num_processes=exec_params["num_processes"],
            total_fail_limit=exec_params["total_fail_limit"],
            total_limit_per_process=exec_params["total_limit_per_process"],
        )
        return

    if choice == 3:
        test_random_rm_core_starts_in_the_end(
            target,
            num_processes=exec_params["num_processes"],
            total_fail_limit=exec_params["total_fail_limit"],
            total_limit_per_process=exec_params["total_limit_per_process"],
            decode_depth=exec_params["decode_depth"],
        )
        return

    if choice == 4:
        test_random_rm_core(
            target,
            num_processes=exec_params["num_processes"],
            total_fail_limit=exec_params["total_fail_limit"],
            total_limit_per_process=exec_params["total_limit_per_process"],
            decode_depth=exec_params["decode_depth"],
        )
        return

    if choice == 5:
        test_txt(
            target,
            num_processes=exec_params["num_processes"],
            total_fail_limit=exec_params["total_fail_limit"],
            total_limit_per_process=exec_params["total_limit_per_process"],
            decode_depth=exec_params["decode_depth"],
        )
        return

    if choice == 6:
        test_for_combined(
            target,
            num_processes=exec_params["num_processes"],
            total_fail_limit=exec_params["total_fail_limit"],
            total_limit_per_process=exec_params["total_limit_per_process"],
        )
        return

    raise ValueError("Unsupported menu choice.")


def main():
    """
    Interactive entry point:
    - ask for function number
    - ask for RM(m, r)
    - ask for execution limits / process count
    - run the selected function
    """
    print_menu()
    choice = read_int("Function number (1-6): ", min_value=1, max_value=6)

    m, r = read_code_params()
    exec_params = read_execution_params(choice)

    target = build_target(choice, m, r)
    run_selected_function(choice, target, exec_params)


if __name__ == "__main__":
    main()