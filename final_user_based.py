import pandas as pd


class UserBasedRecoSystem:
    ut_df = None
    item_ids = None
    user_ids = None
    avg_ratings = None
    num_recos = 10
    k = 5

    def __init__(self, utility_matrix_path, avg_ratings_path):
        UserBasedRecoSystem.ut_df = pd.DataFrame.from_csv(utility_matrix_path, header=0, index_col=0)
        print("Loaded utility matrix")
        UserBasedRecoSystem.item_ids = UserBasedRecoSystem.ut_df.index.tolist()
        UserBasedRecoSystem.user_ids = UserBasedRecoSystem.ut_df.keys().tolist()
        UserBasedRecoSystem.avg_ratings = {}
        lines = [line.rstrip('\n') for line in open(avg_ratings_path)]
        for line in lines:
            line_split = line.split(',')
            UserBasedRecoSystem.avg_ratings[line_split[0]] = float(line_split[1])
        print("Loaded avg ratings")

    def get_users_by_item_rated(self, item_id):
        users = []
        ut_df_transpose = UserBasedRecoSystem.ut_df.T
        item_id_df = ut_df_transpose[item_id]
        for user_id in UserBasedRecoSystem.user_ids:
            if item_id_df[user_id] != 0:
                users.append(user_id)
        return users

    def predict_ratings(self, user_ratings):

        user_u_ratings = {}
        avg_user_u_rating = sum(user_ratings.values()) / len(user_ratings.values())

        for item_id in UserBasedRecoSystem.item_ids:
            user_u_ratings[item_id] = 0

        for item_id in user_ratings.keys():
            user_u_ratings[item_id] = user_ratings[item_id]

        similarities = {}
        for user_id in UserBasedRecoSystem.user_ids:
            user_u_v_df = ((UserBasedRecoSystem.ut_df[user_id]).to_frame())
            user_u_v_df['new_user'] = user_u_ratings.values()

            pcc = user_u_v_df.corr(method='pearson', min_periods=1)[user_id]['new_user']
            similarities[user_id] = pcc

        print("Calculated similarities")
        sorted_user_pcc_list = sorted(similarities, key=similarities.get, reverse=True)

        items_to_be_rated = [x for x in user_u_ratings.keys() if user_u_ratings[x] == 0]

        for item_id in items_to_be_rated:

            users_rated = self.get_users_by_item_rated(item_id)

            k_similar_users = []
            for user in sorted_user_pcc_list:
                if user in users_rated:
                    k_similar_users.append(user)
                    if len(k_similar_users) == UserBasedRecoSystem.k:
                        break

            item_id_prediction_numerator = 0
            item_id_prediction_denominator = 0

            for v_user in k_similar_users:
                sim_u_v = similarities[v_user]
                item_id_prediction_numerator += sim_u_v * (
                        UserBasedRecoSystem.ut_df[v_user][item_id] - UserBasedRecoSystem.avg_ratings[v_user])
                item_id_prediction_denominator += abs(sim_u_v)

            item_id_prediction = item_id_prediction_numerator / item_id_prediction_denominator
            item_id_prediction += avg_user_u_rating

            user_u_ratings[item_id] = item_id_prediction
            print(item_id, item_id_prediction)
        return user_u_ratings

    def get_recommendations(self, user_ratings):
        predicted_ratings = self.predict_ratings(user_ratings)
        recommended_items = sorted(predicted_ratings, key=predicted_ratings.get, reverse=True)
        reco_items = []
        for item in recommended_items:
            if item not in user_ratings.keys():
                reco_items.append(str(item)) 
                if len(reco_items) == UserBasedRecoSystem.num_recos:
                    break
        return reco_items
