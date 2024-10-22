import RM
matrix = RM.generate_matrix()
print(matrix)
c=RM.RM()
message = [0,1,1,0,1,0,1,1,0,1,0]  # 11 бит
print(c.Encoding42(message))
emessage=c.Encoding42(message)
emessage[2]=1
print(c.Decoding42(emessage))