import pandas as pd
import numpy
import numpy.ma as ma
import math
import time

def reduceCols(A, startIndex, numCols):
    m = len(A)
    n = len(A[0])
    B = []
    for i in range(m):
        B.append(A[i][startIndex:(startIndex+numCols)])
    return B

def reduceRows(A, k):
    B = A[:k]
    return B

def norm(v):
    s = 0
    for i in range(len(v)):
        s = s + (v[i] * v[i])
    n_v = math.sqrt(s)
    return n_v

def normalise(A, col, n_col):
    m = len(A)
    #print(A)
    for i in range(m):
        A[i][col] = A[i][col] / n_col

def diag(entries, numRows, numCols):
    D = []
    for i in range(numRows):
        row = []
        for j in range(numCols):
            if j == i:
                row.append(entries[i])
            else:
                row.append(0)
        D.append(row)
    return D

def getIdentity(numRows, numCols):
    entries = [1] * min(numRows, numCols)
    I = diag(entries, numRows, numCols)
    return I

def copy(M):
    C = M.view()
    return C

def svd(A, mRow, nCol, kRank):
    #U = reduceCols(copy(A), kRank)
    #V = getIdentity(nCol, kRank)
    start_time = time.time()
    U = copy(A)
    V = getIdentity(nCol, nCol)
    tol = math.exp(-8)
    converge = tol + 1

    j_time = time.time()
    while converge > tol:
        print('converge', converge)
        for j in range(1, nCol):
            for i in range(kRank):
                #print('i', i)
                alpha = 0
                beta = 0
                gamma = 0
                #for k in range(kRank):
                for k in range(mRow):
                    alpha = alpha + ((U[k][i])**2)
                    beta = beta + ((U[k][j])**2)
                    gamma = gamma + (U[k][j] * U[k][i])
                #print(alpha)
                #print(gamma)
                #print(beta)
                secargden = math.sqrt(alpha * beta)
                if secargden != 0:
                    converge = min(converge, abs(gamma) / math.sqrt(alpha * beta))
                t = 0
                if gamma != 0:
                    zeta = (beta - alpha) / (2 * gamma)
                    sign = 0
                    if zeta != 0:
                        sign = int(zeta / abs(zeta))
                    t = sign / (abs(zeta) + math.sqrt(1 + (zeta * zeta)))
                else:
                    t = 0
                c = 1 / math.sqrt(1 + (t * t))
                s = c * t
                #for k in range(kRank):
                for k in range(mRow):
                    t = U[k][i]
                    U[k][i] = (c * t) - (s * U[k][j])
                    U[k][j] = (s * t) + (c * U[k][j])
                for k in range(nCol):
                    t = V[k][i]
                    V[k][i] = (c * t) - (s * V[k][j])
                    V[k][j] = (s * t) + (c * V[k][j])
            if ((j+1)%100) == 0:
                print('j is', j)
                next_time = time.time()
                print('time taken by last 100 iterations is', (next_time - j_time), 'seconds')
                j_time = next_time
    singvals = [0] * kRank
    #singvals = [0] * nCol
    for j in range(kRank):
    #for j in range(nCol):
        Uj = []
        for i in range(mRow):
            Uj.append(U[i][j])
        singvals[j] = norm(Uj)
        normalise(U, j, singvals[j])        
        #print(Uj)
        #print(singvals[j])
    #print(singvals)
    #singvals = sorted(singvals).reverse()
    #print(len(singvals))
    #print(kRank)
    S = diag(singvals, kRank, kRank)
    #S = diag(singvals, nCol, nCol)
    #print(kRank)
    #U = numpy.where(numpy.max(U, axis=0)==0, U, U*1./numpy.max(U, axis=0))
    U = reduceCols(U, 0, kRank)
    V = reduceCols(V, 0, kRank)
    #final_U = transpose(final_U)
    #final_V = transpose(final_V)
    print('Time taken to compute SVD was', (time.time() - start_time), 'seconds')
    return U, S, V
    
def rmse(M, N):
    numRows = len(M)
    numCols = len(M[0])
    total = 0
    ctr = 0
    for i in range(numRows):
        for j in range(numCols):
            diff = M[i][j] - N[i][j]
            sq = diff * diff
            total = total + sq
            ctr += 1
    mean = total/ctr
    error = math.sqrt(mean)
    return error

def writeToFile(U, V):
    numpy.savetxt('user_vectors.csv', U, delimiter=',')
    numpy.savetxt('item_vectors.csv', V, delimiter=',')

def readFromFile():
    #A = randomMatrix(100)
    df = pd.read_csv('utility_matrix_ut.csv', skiprows=1, index_col=False, header=None)
    df.drop(df.columns[0], axis=1, inplace=True)
    A = df.values
    #print(A[0])
    return A

numpy.seterr(divide='ignore', invalid='ignore')

raw = readFromFile()
col_avg = numpy.mean(raw, axis=0)
A = numpy.where(raw==0, ma.array(raw, mask=(raw==0)).mean(axis=0), raw)
A = A - A.mean(axis=1, keepdims=True)
m = len(A)
n = len(A[0])
print('Generated')
k = 14
U, S, V = svd(A, m, n, k)
#print('Dimensions of U are', len(U), len(U[0]))
#print('Dimensions of S are', len(S), len(S[0]))
T = numpy.transpose(V)
#print('Dimensions of V_T are', len(T), len(T[0]))
P = numpy.dot(numpy.dot(U, S), T)
#print('Product computed. Dimensions are', len(P), len(P[0]))
print('Error is', rmse(P, A))
writeToFile(U, V)
