from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
# from scipy.sparse._csr import csr_matrix

from recommender.BaseRecommender import BaseRecommender

class ContentBasedRecommender(BaseRecommender):
    _tfidf:TfidfVectorizer
    _tfidf_matrix: Any

    def _validate_data(self, data: pd.DataFrame):
        assert 'item_id' in data.columns and \
        'description' in data.columns and \
        'cost' in data.columns , 'Not valid'

    def fit(self,data:pd.DataFrame,description_column:str='description'):
        super().fit(data)
        
        self._tfidf=TfidfVectorizer(stop_words='english')
        self._tfidf_matrix=self._tfidf.fit_transform(self.data[description_column])
    

    def recommend_1(self, query:str,limit=10)->pd.DataFrame:
        query_to_vector=self._tfidf.transform([query.lower()])

        cosine_distance_to_data=linear_kernel(query_to_vector,self._tfidf_matrix)[0]

        candidate_indices=cosine_distance_to_data.argsort()[::-1][:limit]

        return self.data.iloc[candidate_indices][['item_id','description']]



        
