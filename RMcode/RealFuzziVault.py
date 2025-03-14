import RMCore
import numpy as np
import hashlib
def sha256_hash(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    elif isinstance(data, int):
        data = str(data).encode('utf-8')
    elif isinstance(data, list):
        data = str(data).encode('utf-8')
    elif not isinstance(data, bytes):
        raise TypeError(f"Неподдерживаемый тип данных для хеширования: {type(data)}")

    return hashlib.sha256(data).hexdigest()
def invert_bits(binary_array, indices):
    binary_array = np.array(binary_array, dtype=np.uint8)
    indices = np.array(indices, dtype=np.int32)
    valid_indices = indices[(indices >= 0) & (indices < len(binary_array))]
    binary_array[valid_indices] ^= 1
    return binary_array
def evaluate_count_of_flipped(m,r):
    #TODO
    return 1
class FuzziVault():
    def __init__(self, m, r, attempts_count):
        self.code=RMCore.RMCore(m,r)
        self.attempts_count=attempts_count
        self.is_locked=False
        self.kee_length=evaluate_count_of_flipped(m,r)
        print("Your vault has been created \n Kee length must be "+str(self.kee_length)+"\n")
    def lock(self, secret, kee):
        if self.is_locked:
            print("Already locked \n")
        else:
            if(len(kee)!=self.kee_length):
                print("Wrong kee length, try again \n")
                exit()
            self.hash=sha256_hash(secret)
            self.vault=invert_bits(self.code.encode(secret), kee)
            self.is_locked=True
            print("Vault has been locked \n")
    def unlock(self, kee):
        cur_vault=invert_bits(self.vault,kee)
        decoded = self.code.real_final_version_decode(cur_vault)
        if(self.hash==sha256_hash(decoded)):
            print("Vault has been unlocked!!! \n ")
        else:
            self.attempts_count-=1
            print("Vault hasn't been unlocked, now u have only " + str(self.attempts_count)+ " attempts \n")



