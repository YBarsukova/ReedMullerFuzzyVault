from math import comb

import RMCore

def guess_mistakes(core, min_poss):
    sum=0
    with open("tuning.txt", "a", encoding="utf-8") as file:
        file.write('test for ' +str(core.code.m) +"  "+ str(core.code.r)+ '\n')
    for t in range(1, (core.code.n//2)):#кол-во внесенных ошибок
        for s in range(1,t):
            p=comb(t,s)/comb(core.code.n, s)
            if (p<min_poss):
                #print(p)
                with open("tuning.txt", "a", encoding="utf-8") as file:
                    file.write('t = '+str(t) +' s = '+str(s)+ '\n')

def guess_real(core, min_poss):
    zn=comb(core.code.n, core.code.k)
    with open("tuning2.txt", "a", encoding="utf-8") as file:
        file.write('test for ' +str(core.code.m) +"  "+ str(core.code.r)+ '\n')
    for t in range(1, (core.code.n//2)):
        p=comb(core.code.n-t, core.code.k)/zn
        if (p<min_poss):
            #print(p)
            with open("tuning2.txt", "a", encoding="utf-8") as file:
                file.write('t = '+str(t) +'\n')