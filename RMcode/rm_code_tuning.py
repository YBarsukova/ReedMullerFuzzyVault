from math import comb

import RMCore

def guess_mistakes(core, min_poss):
    sum=0
    with open("tuning.txt", "a", encoding="utf-8") as file:
        file.write('test for ' +str(core.code.m) +"  "+ str(core.code.r)+ '\n')
    for t in range(1, core.code.n-1):#кол-во внесенных ошибок
        for s in range(1,t):
            p=comb(t,s)/comb(core.code.n, s)
            if (p<min_poss):
                with open("tuning.txt", "a", encoding="utf-8") as file:
                    file.write('t = '+str(t) +' s = '+str(s)+ '\n')