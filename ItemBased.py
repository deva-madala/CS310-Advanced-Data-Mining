import  pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
import numpy as np

class recco:

    def __init__(self):
        self.a = np.load('sim.npy')
        self.listOfDictionaries = np.load('lod.npy')


    def ItemBasedCF(self,inp):
        user1 = [None]*3706
        user_books = []
        indexMap = {}
        reverseIndexMap = {}
        ptr=0
        testdf = pd.read_csv('ratings.csv')
        avg = pd.read_csv('User_Average.csv')
        book = pd.read_csv('movies.csv',encoding='latin-1')
        bookdict=book.set_index('movie_id').T.to_dict('list')
        avgdict = avg.set_index('user_id').T.to_dict('list')
        testdf=testdf[['user_id','rating']].groupby(testdf['movie_id'])

        count=1
        for groupKey in testdf.groups.keys():
                        #groupDF = testdf.get_group(groupKey)
                        #t = groupDF.set_index('user_id').T.to_dict('list')
                        #for k,v in t.items():
                         #       t[k]= t[k][0]-avgdict[k][1]
                        indexMap[ptr]=groupKey
                        reverseIndexMap[groupKey] = ptr
                        ptr=ptr+1
                        count+=1
                        #listOfDictionaries.append(t)


        for k,v in inp.items():
            user_books.append(reverseIndexMap[k])
            user1[reverseIndexMap[k]]=v

        #dictVectorizer = DictVectorizer(sparse=True)
        #vector = dictVectorizer.fit_transform(listOfDictionaries)
        #pairwiseSimilarity = cosine_similarity(vector)
        sim = self.a
        '''print('generating cosine matrix')
        for i in user_books:
               b = sim[i]
               for d,j in enumerate(b):
                                sum1 = 0
                                sum2 = 0
                                sum3 = 0
                                sum4 = 0
                                diff1 = self.listOfDictionaries[i].keys()-self.listOfDictionaries[d].keys()
                                diff2 = self.listOfDictionaries[d].keys() - self.listOfDictionaries[i].keys()
                                for k,v in self.listOfDictionaries[i].items():
                                    sum1+=v**2
                                    if k not in diff1 and k not in diff2:
                                            sum2+=v**2
                                for m,n in self.listOfDictionaries[d].items():
                                    sum3+=n**2
                                    if m not in diff1 and m not in diff2:
                                            sum4+=n**2
                                s1 = math.sqrt(sum1)
                                s2 = math.sqrt(sum2)
                                s3 = math.sqrt(sum3)
                                s4 = math.sqrt(sum4)
                                sim[i][d] = (sim[i][d]*s1*s3)/(s2*s4)


'''

        top_n =[]
        for i in user_books:
            top_n.append(sim[i])
        z = [sum(x) for x in zip(*top_n)]
        for i in user_books:
            z[i]=0
        top = np.array(z)
        k = top.argsort()[::-1][:20]

        print('Similarity matrix loaded')


        for o,x in enumerate(user1):
                if x!=None:
                        x = (2*(x-1)-4)/4
                        user1[o]=x

        predict = []
        order = []
        for count1,i in enumerate(user1):
                if i==None:
                    if count1 in k:
                        order.append(count1)
                        sum1 = 0
                        sum2 = 0
                        f = count1
                        for count2,j in enumerate(user1):
                                g = count2
                                if j!=None:
                                        sum1+=(sim[g][f]*user1[g])
                                        sum2+=abs(sim[g][f])

                        p = sum1/sum2
                        prediction = ((p+1)*2)+1
                        predict.append(prediction)



        recommend = np.array(predict)
        ind=[]
        if len(set(predict)) == 1:
            for i in range(10):
                ind.append(i)
            flag =1
        else:
            ind = recommend.argsort()[::-1][:10]
            flag = 0
        print('Recommendations for you based on movies you have rated:')
        mov = list(k)
        reccos =[]
        for i in ind:
                if flag==1:
                    a = mov[i]
                else:
                    a = order[i]
                bk = indexMap[a]
                k = bookdict[bk]
                print(k)
                reccos.append(str(bk))
        return reccos

