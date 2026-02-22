from math import comb


def guess_mistakes(core, min_poss):
    """
    Write (t, s) pairs to tuning.txt where C(t, s) / C(n, s) < min_poss for the given core.code.
    """
    with open("../out/tuning.txt", "a", encoding="utf-8") as file:
        file.write("test for " + str(core.code.m) + "  " + str(core.code.r) + "\n")

    for t in range(1, core.code.n // 2):  # number of injected errors
        for s in range(1, t):
            p = comb(t, s) / comb(core.code.n, s)
            if p < min_poss:
                with open("../out/tuning.txt", "a", encoding="utf-8") as file:
                    file.write("t = " + str(t) + " s = " + str(s) + "\n")


def guess_real(core, min_poss):
    """
    Write t values to tuning2.txt where C(n - t, k) / C(n, k) < min_poss for the given core.code.
    """
    zn = comb(core.code.n, core.code.k)

    with open("../out/tuning2.txt", "a", encoding="utf-8") as file:
        file.write("test for " + str(core.code.m) + "  " + str(core.code.r) + "\n")

    for t in range(1, core.code.n // 2):
        p = comb(core.code.n - t, core.code.k) / zn
        if p < min_poss:
            with open("../out/tuning2.txt", "a", encoding="utf-8") as file:
                file.write("t = " + str(t) + "\n")