from flask import Flask, render_template, request
import final_user_based
import ItemBased
import svd_recommend

app = Flask(__name__)

user_based_sys = final_user_based.UserBasedRecoSystem('utility_matrix.csv', 'avg_rating.csv')
item_based_sys = ItemBased.recco()


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/recommendations', methods=['POST', 'GET'])
def result():
    result = request.form
    dic = {}
    f = open("movies.dat")
    s = f.read()
    f.close()
    s = s.split('\n')
    for each in s[:-1]:
        a = each.split("::")
        dic[a[1]] = a[0]
    names = []
    ratings = []
    ids = []
    for i in range(1, 6):
        names.append(result['m' + str(i)])
        ratings.append(result['v' + str(i)])
    for i in range(0, 5):
        if names[i] in dic:
            ids.append(dic[names[i]])
        else:
            return render_template("error.html")
    if len(names) != len(set(names)):
        return render_template("error.html")
    dicb = {}
    for i in range(0, 5):
        dicb[int(ids[i])] = int(ratings[i])
    # print(dicb)

    finalids1 = item_based_sys.ItemBasedCF(dicb)
    finalnames1 = []
    finalids2 = user_based_sys.get_recommendations(dicb)
    finalnames2 = []
    finalids3 = svd_recommend.guessRatings(dicb)
    finalnames3 = []

    for i in range(0, 10):
        for each in dic:
            if dic[each] == finalids1[i]:
                finalnames1.append(each)
            if dic[each] == finalids2[i]:
                finalnames2.append(each)
            if dic[each] == finalids3[i]:
                finalnames3.append(each)

    return render_template("recommendations.html", film1=finalnames1, film2=finalnames2, film3=finalnames3)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
