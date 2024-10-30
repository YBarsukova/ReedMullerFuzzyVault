import RM
import random
c=RM.RM(10,1)
for i in range(10):
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
