# ReedMullerFuzzyVault

Experimental repository for working with **Reed–Muller codes** and related decoding / fuzzy-vault ideas.

## Overview

This repository contains:

- **Code implementations**
  - `ReedMuller` (classic encoder/decoder)
  - `RMCore` (custom/recursive decoding logic)
  - `CombinedRM` (optimized / alternative implementation)
  - fuzzy vault related modules

- **Testing / experiment utilities**
  - Monte Carlo probability tests
  - parallel workers (`ProcessPoolExecutor`)
  - filter experiments and result logging

- **Output helpers**
  - writing results to **Excel** (`.xlsx`)
  - writing results to **text files** (`.txt`)

- **Interactive CLI (`main.py`)**
  - asks for code parameters `RM(m, r)`
  - asks which test function to run
  - asks execution settings (processes / limits)
  - runs the selected experiment and saves results

---

## Quick start

Run from the project root:

```bash
python main.py
````

You will then choose one of the available functions (`1..6`), enter `m`, `r`, and additional runtime parameters.

---

## Notes before running

* `m >= 1`
* `0 <= r <= m`
* `total_fail_limit` must be divisible by `num_processes`
* Result files are saved into the `out/` directory

---

# Main functions (CLI menu)

Below is a short description of each function available in `main.py`, plus a sample launch for each.

---

## 1) `test_random(code)`

**What it does:**
Runs parallel Monte Carlo tests for the **classic `ReedMuller` decoder** and saves results to Excel.

**Output file:**

* `out/test_data.xlsx`

### Sample launch

```text
Function number (1-6): 1

Enter Reed-Muller code parameters RM(m, r):
m (>= 1): 6
r (0..6): 2

Number of parallel processes [Enter = 4]: 8
Total fail limit [Enter = 1024]: 512
```

---

## 2) `test_random2(code)`

**What it does:**
Same idea as function 1, but uses an **early-break Monte Carlo variant** (usually faster / more practical).

**Output file:**

* `out/test_data.xlsx`

### Sample launch

```text
Function number (1-6): 2

Enter Reed-Muller code parameters RM(m, r):
m (>= 1): 6
r (0..6): 2

Number of parallel processes [Enter = 4]: 8
Total fail limit [Enter = 1024]: 512
Per-process trial limit (total_limit_per_process) [Enter = 100000]: 50000
```

---

## 3) `test_random_rm_core_starts_in_the_end(core)`

**What it does:**
Runs Monte Carlo tests for **`RMCore`**, scanning error counts **from high to low** (`n//2 -> 1`).
If success becomes very high, the remaining lower counts may be filled automatically.

**Output file:**

* `out/test_recursed.xlsx`

### Sample launch

```text
Function number (1-6): 3

Enter Reed-Muller code parameters RM(m, r):
m (>= 1): 6
r (0..6): 2

Number of parallel processes [Enter = 60]: 8
Total fail limit [Enter = 1020]: 512
Per-process trial limit (total_limit_per_process) [Enter = 100000]:
RMCore decode depth (decode_depth) [Enter = 3]:
```

---

## 4) `test_random_rm_core(core)`

**What it does:**
Runs the standard upward Monte Carlo scan for **`RMCore`** (from `mistakes_count + 1` upward until success rate reaches zero).

**Output file:**

* `out/test_recursed.xlsx`

### Sample launch

```text
Function number (1-6): 4

Enter Reed-Muller code parameters RM(m, r):
m (>= 1): 6
r (0..6): 2

Number of parallel processes [Enter = 60]: 8
Total fail limit [Enter = 1020]: 512
Per-process trial limit (total_limit_per_process) [Enter = 100000]:
RMCore decode depth (decode_depth) [Enter = 3]:
```

> Note: depending on the current implementation, some `RMCore` recursive paths may be unstable for certain parameter combinations (especially small `r`, e.g. `r = 1`).

---

## 5) `test_txt(core)`

**What it does:**
Runs `RMCore` tests and appends results to a text file (plus a timestamp at the end).

**Output file:**

* `out/results.txt`

### Sample launch

```text
Function number (1-6): 5

Enter Reed-Muller code parameters RM(m, r):
m (>= 1): 6
r (0..6): 2

Number of parallel processes [Enter = 60]: 8
Total fail limit [Enter = 1020]: 512
Per-process trial limit (total_limit_per_process) [Enter = 100000]:
RMCore decode depth (decode_depth) [Enter = 3]:
```

---

## 6) `test_for_combined(core)`

**What it does:**
Runs tests for **`CombinedRM`** and writes results to a text file.
It scans error counts with a coarse step (`-10`) for faster exploration.

**Output file:**

* `out/results_for_majory.txt`

### Sample launch

```text
Function number (1-6): 6

Enter Reed-Muller code parameters RM(m, r):
m (>= 1): 5
r (0..5): 2

Number of parallel processes [Enter = 60]: 8
Total fail limit [Enter = 1020]: 512
Per-process trial limit (total_limit_per_process) [Enter = 100000]:
```


