from typing import Any,List,Union
from datetime import datetime,time,timedelta

import pandas as pd
from pyspark.ml.fpm import FPGrowth,FPGrowthModel
from pyspark.context import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql.window import Window
from pyspark.sql import functions as F

from recommender.utils import get_config
from recommender.BaseRecommender import BaseRecommender

class FpGrowthRecommender(BaseRecommender):
    spark:SparkContext
    _model:FPGrowthModel

    def __init__(self,spark:SparkContext) -> None:
        super().__init__()
        self.spark=spark

    def _validate_data(self, data: DataFrame):
        assert 'user_id' in data.columns and \
        'items' in data.columns and \
        'timestamp' in data.columns , 'Not valid'
    
    def fit(self, data:DataFrame,dt:datetime):
        super().fit(data)
        try:
            model=FPGrowthModel.load(
                get_config('PURCHASE_FPGROWTH','model_path').replace('{{yyyymm}}',dt.strftime('%Y%m'))
            )
            print('load')
        except:
            fp=FPGrowth(minConfidence=0.001,minSupport=0.001,itemsCol='items')
            model=fp.fit(data)
            print('fit')

            model.write().overwrite().save(get_config('PURCHASE_FPGROWTH','model_path').replace('{{yyyymm}}',dt.strftime('%Y%m')))

        self._model=model
        
    def retrain(self, new_data: Union[pd.DataFrame, DataFrame],dt:datetime=datetime.now()):
        self.fit(new_data,dt)

    def recommend_1(self, user_id: int):
        pass
    
    def get_latest_transaction_of_users(self,user_ids:List[int]):
        window=Window.partitionBy('user_id').orderBy(F.desc('timestamp'))
        users_df=self.spark.createDataFrame(
            pd.DataFrame(
                {
                    'user_id':user_ids
                }
            )
        )
        user_latest_trans=self.data.join(users_df,on='user_id')\
            .withColumn('order_trans_inverse',F.row_number().over(window))\
            .filter(F.col('order_trans_inverse')==1)\
            .select('user_id',F.col('items').alias('prev_package'))
        
        return user_latest_trans

    def recommend_batch(self,user_ids:List[int],n_recommends:int=10,prev_purchased_col:str='prev_package')->DataFrame:
        data_input=self.get_latest_transaction_of_users(user_ids=user_ids)

        return self._model.transform(
            dataset=data_input.select(
                'user_id',
                F.col(prev_purchased_col).alias('items')
            )
        ).select('user_id',F.slice('prediction',1,n_recommends).alias('recommendation')).toPandas()
