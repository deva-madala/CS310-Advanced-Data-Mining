f=open("movies.dat",'r')
s=f.read()
f.close()
s=s.split('\n')
k=[]
for each in s[:-1]	:
	k.append(each.split("::")[1])
f=open("moviesnames.txt","w")
f.write(str(k))
f.close()
#print(s)