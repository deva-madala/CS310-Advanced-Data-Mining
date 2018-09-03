import numpy
import numpy.ma as ma
import pandas as pd
import operator

def readData():
    hdr = pd.read_csv('utility_matrix_ut.csv', nrows=1, header=None)
    df = pd.read_csv('utility_matrix_ut.csv', skiprows=1, index_col=False, header=None)
    df.drop(df.columns[0], axis=1, inplace=True)
    #print(hdr.values)
    movies = hdr.values[0][1:]
    A = df.values
    #print(A[0])
    return A, movies

def readUsers():
    df = pd.read_csv('user_vectors.csv', header=None)
    U = df.values
    #print(A[0])
    return U

def readItems():
    df = pd.read_csv('item_vectors.csv', header=None)
    V = df.values
    #print(A[0])
    return V

def normalise(E, row):
    Au = numpy.append(E, [row], axis=0)
    #print(len(Au[0]))
    Au = numpy.where(Au==0, ma.array(Au, mask=(Au==0)).mean(axis=0), Au)
    #print(len(Au[0]))
    Au = Au - Au.mean(axis=1, keepdims=True)
    Au = Au[-1]
    return Au

def guessRatings(new_ratings):
    A, movie_ids = readData()
    U = readUsers()
    V = readItems()
    T = numpy.transpose(V)
    n = len(movie_ids)
    #print(len(movie_ids))
    #for i in range(n):
     #   print(movie_ids[i])
    new_user = [new_ratings.get(int(movie_ids[i]), 0) for i in range(n)]
    #new_user = 5 * numpy.random.random(size=(1, n))
    #Pu = P + new_user
    norm_user = normalise(A, new_user)
    Uku = numpy.dot(norm_user, V)
    Aku = numpy.dot(Uku, T)
    reco = []
    high_ID = dict()
    for i in range(n):
        if new_user[i]==0:
            high_ID[int(movie_ids[i])]= int(Aku[i])
    sorted_x = sorted(high_ID.items(), key=operator.itemgetter(1))
    sorted_x.reverse()
    for i in range(10):
        reco.append(str(sorted_x[i][0]))
    #print(reco)
    #print(predictions[0])
    #print([int(x) for x in pred_sorted])
    #print([int(x) for x in predictions[0]])
    #print([int(x) for x in new_user[0]])
    return reco

#test_user = {5: 5, 6: 1, 7: 3, 8: 4}
#print(guessRatings(test_user))
