from typing import Any,List

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


from recommender.BaseRecommender import BaseRecommender


class CollaborativeFilteringRecommender(BaseRecommender):

    def _validate_data(self, data: pd.DataFrame):
        assert 'user_id' in data.columns and \
        'item_id' in data.columns and \
        'score' in data.columns , 'Not valid'

    def fit(self, data: pd.DataFrame):
        super().fit(data)

        customer_item_matrix=pd.pivot(data,index='user_id',columns='item_id',values='score',)\
            .applymap(lambda x: 1 if x > 0 else 0)

        user_user_sim_matrix = pd.DataFrame(
            cosine_similarity(customer_item_matrix)
        )

        user_user_sim_matrix.columns=customer_item_matrix.index
        user_user_sim_matrix.index=customer_item_matrix.index

        self.user_user_sim_matrix=user_user_sim_matrix
        self.customer_item_matrix=customer_item_matrix

    

    def recommend_1(self, user_id: int,k_near:int=5,n_recommends:int=10)->List[int]:
        sim_users=self.user_user_sim_matrix.loc[user_id].sort_values(ascending=False)

        top_k_user_id=sim_users[1:k_near].index
        top_k_sim_score=sim_users[1:k_near].values

        items_bought_by_A = (self.customer_item_matrix.loc[user_id].iloc[
            self.customer_item_matrix.loc[user_id].to_numpy().nonzero()
        ].tolist())

        recommend_items=(
            self.customer_item_matrix
            .loc[lambda x:x.index.isin(top_k_user_id)]
            .mul(top_k_sim_score,axis='index')
            .sum(axis=0).sort_values(ascending=False)
            [:n_recommends+len(items_bought_by_A)].index.tolist()
        )

        return [x for x in recommend_items if x not in items_bought_by_A][:n_recommends]

    def recommend_batch(self,users_ids:List[int],k_near:int=5,n_recommends:int=10)->pd.DataFrame:
        res_pd=pd.DataFrame(
            {
                'user_id':users_ids
            }
        )

        recommendations=[]
        for user_id in users_ids:
            recommendations.append(
                self.recommend_1(user_id=user_id,k_near=5,n_recommends=10)
            )
        
        res_pd=res_pd.assign(recommendation=recommendations)
        return res_pd

    
        



        