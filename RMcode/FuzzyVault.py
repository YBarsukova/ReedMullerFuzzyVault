from copy import copy

import RM
import random


def random_set(count, min, max):
    unique_numbers = set()
    while len(unique_numbers) < count:
        number = random.randint(min, max)
        unique_numbers.add(number)
    return unique_numbers

def missing_numbers_in_range(min_value, max_value, us_set):
    full_set = set(range(min_value, max_value + 1))
    missing_numbers = full_set - us_set
    return missing_numbers

class Vault:
    def __init__(self,m,r):
        self.code=RM.RM(m,r)
        self.attempts_count=3

    def filling_trash(self, enc, count):
        rs=random_set(count, 0,  self.code.n-1)
        self.kee=missing_numbers_in_range(0, self.code.n-1, rs)
        for ind in rs:
            enc[ind]=enc[ind]^1
        return enc

    def filling_erases(self, vault, ints):
        for m in missing_numbers_in_range(0, self.code.n-1, ints):
            vault[m]=3
        return vault

    def lock(self, data_input):
        self.secret=data_input
        enc=self.code.encode(data_input)
        self.vault=self.filling_trash(enc, self.code.erases_count-2)
        print(f"{self.kee} - yours password \n")

    def unlock(self, ints):
        print("va ",self.vault )
        if self.attempts_count>0:
            attempt=self.filling_erases(copy(self.vault), ints)
            dec=self.code.decode(attempt)
            if (dec==self.secret):
                print("Unlocked!!!")
            else:
                print("try again")
                self.attempts_count-=1
                print(self.attempts_count)
        else:
            print("There is no attempts")