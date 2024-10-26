import RM
import random
#from RMcode.RM import sum_binomial, fill_bit_array, MatrixGenerator, Generation_Gr, Creation_G1
import numpy as np
from itertools import combinations

import RMcode.RM
from RMcode.RM import sum_binomial

# c=RM.RM(4,2)
# message = [0,1,1,1,1,1,0,1,1,0,0]  # 11 бит
# print(c.encode(message))
# emessage=c.encode(message)
# #emessage[2]=1
# print("dec", c.decode2(emessage))

# print(RM.get_multipliers(2, 3,2))
# ##matrix = RM.generate_matrix()
# ##print(matrix)
# #print(RMcode.RM.sum_binomial(5,2))
c=RM.RM(4,2)
for i in range(100):
    message = [random.choice([0,1]) for i in range(RM.sum_binomial(c.m, c.r))]  # 11 бит
#print(c.encode(message))
    emessage=c.encode(message)
    emessage[2]=emessage[2]^1
    d = c.decode2(emessage)
    assert message == d
print(c.decode2(emessage))
##print(sum_binomial(4,2))
##print(MatrixGenerator(4,2))
##print(Generation_Gr(Creation_G1(4),2))
#print(RM.Creation_G1(3))