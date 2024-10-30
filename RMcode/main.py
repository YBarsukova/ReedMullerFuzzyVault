import time
import random
import RM

def measure_main_execution_time():
    start_time = time.time()
    c=RM.RM(4,2)
    for i in range(200000):
        message = [random.choice([0,1]) for i in range(RM.sum_binomial(c.m, c.r))]
        emessage=c.encode(message)
        #print(emessage)
        emessage[2]=emessage[2]^1
        for j in range(3):
            emessage[j] = 3
        #print(emessage)
        d = c.Decode(emessage)
        #print(d)
        assert message == d
    print(c.decode2(emessage))
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
measure_main_execution_time()
