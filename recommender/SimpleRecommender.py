from typing import Any,List

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


from recommender.BaseRecommender import BaseRecommender

class SimpleRecommender(BaseRecommender):
    _ranking:pd.DataFrame
    
    def ranking(self):
        self._ranking=self.data.groupby('item_id').agg({
            'user_id':pd.Series.nunique
            }
        ).rename(columns={'user_id':'num_distinct_user'}).sort_values(['num_distinct_user'],ascending=False).reset_index()

    def fit(self, data: pd.DataFrame):
        super().fit(data)
        self.ranking()

    def recommend_1(self,n_recommends:int=10):
        return self._ranking.item_id.values[:n_recommends]