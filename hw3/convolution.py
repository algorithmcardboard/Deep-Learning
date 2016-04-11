import numpy as np

I = np.array([4, 5, 2, 2, 1, 3, 3, 2, 2, 4, 4, 3, 4, 1, 1, 5, 1, 4, 1, 2, 5, 1, 3, 1, 4]).reshape(5,5)

K = np.array([4, 3, 3, 5, 5, 5, 2, 4, 3]).reshape(3,3)

N = I.shape[0]
m = K.shape[0]

output = np.zeros((N-m+1, N-m+1))

columns = np.array([[0, 0, 0],
        [1, 1, 1],
        [2, 2, 2]])

rows = np.array([[0, 1, 2],
        [0, 1, 2],
        [0, 1, 2]])

for i in range(0, N - m + 1):
    for j in range(0, N -m ++ 1):
        subarray = I[columns+i, rows+j]
        #print(subarray)
        output[i][j] = np.sum((subarray * K))

print(output)
"""
print(I)
print("\n")
print(output)
"""
