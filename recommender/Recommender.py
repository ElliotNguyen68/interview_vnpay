from typing import Any,List
import os

os.environ["CONFIG_PATH"] = "/home/locnt7"

os.environ["SPARK_SERVER"] = "SPARK@95"

from pdp.core import hdfs_utils, spark_utils
from pyspark.context import SparkContext
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from datetime import datetime

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from recommender.BaseRecommender import BaseRecommender
from recommender.AssociationRule import FpGrowthRecommender
from recommender.ContentBasedRec import ContentBasedRecommender
from recommender.SimpleRecommender import SimpleRecommender
from recommender.CollaborativeFilteringRec import CollaborativeFilteringRecommender



class Recommender(BaseRecommender):
    data_parquet:DataFrame
    spark:SparkContext
    
    _cf_model:CollaborativeFilteringRecommender
    _content_model:ContentBasedRecommender
    _association_rule:FpGrowthRecommender
    _simple_recommender:SimpleRecommender

    def init_spark(self,app_name:str='Product recommendation')->SparkContext:
        NO_CORES=8
        spark = spark_utils.get_spark_session(
            app_name=app_name,
            cores=NO_CORES,
            driver_memory=32,
            executor_memory=32,
            ui_port=9696,
            log_level="ERROR",
        )
        return spark

    def __init__(self,spark:SparkContext=None) -> None:
        super().__init__()
        if not spark:
            self.spark=self.init_spark()
        else:
            self.spark=spark


    def _validate_data(self, data: pd.DataFrame):
        assert 'user_id' in data.columns and \
        'item_id' in data.columns and \
        'timestamp' in data.columns and \
        'trans_id' in data.columns and \
        'amount' in data.columns and \
        'cost' in data.columns and \
        'description' in data.columns, 'Not valid'

    def fit(self, data: pd.DataFrame):
        super().fit(data)
        self.data_parquet=self.spark.createDataFrame(self.data)
    
        self._checkin_user_data=self.data_parquet.groupBy('user_id').agg(
            F.countDistinct('trans_id').alias('num_trans')
        ).cache()

    def recommend_1(self, user_id: int):
        pass

    def _init_recommendation(self,data_item:pd.DataFrame,dt:datetime=datetime.now()):

        self._content_model=ContentBasedRecommender()
        self._content_model.fit(data_item)

        self._cf_model=CollaborativeFilteringRecommender()

        data_cf=self.data.groupby(['user_id','item_id']).agg(
            {
                'amount':sum
            }
        ).reset_index().rename(columns={'amount':'score'}).drop_duplicates()
        self._cf_model.fit(
           data_cf
        )

        self._association_rule=FpGrowthRecommender(spark=self.spark)
        data_fp=self.data_parquet.groupBy('user_id','trans_id','timestamp').agg(
            F.collect_set('item_id').alias('items')
        )
        self._association_rule.fit(data=data_fp,dt=dt)

        self._simple_recommender=SimpleRecommender()
        self._simple_recommender.fit(self.data)


    def _segment_users(self,user_ids:List[int])->pd.DataFrame:
        users_df=self.spark.createDataFrame(
            pd.DataFrame(
                {
                    'user_id':user_ids
                }
            )
        )

        type_users=users_df.join(
            self._checkin_user_data,on='user_id',how='left'
        ).withColumn('segment',F.when(F.col('num_trans')>3,'old').otherwise(
            F.when((F.col('num_trans')<=3)&(F.col('num_trans')>=1),'warm').otherwise('new')
            )
        ).select('user_id','segment').toPandas()
        return type_users
    

    def recommend_batch(self,user_ids:List[int],n_recommends:int=10):
        segment_pd=self._segment_users(user_ids)

        num_new=0
        num_warm=0
        num_old=0


        ################
        # New          #
        ################
        
        new_users=segment_pd.loc[lambda x:x.segment=='new']
        num_new=len(new_users)
        if num_new:
            most_frequent_item=self._simple_recommender.recommend_1(n_recommends=n_recommends)

            new_users['recommendation']=new_users.apply(lambda _:most_frequent_item,axis=1) 

            recommendation_new_users=new_users

        ###############
        # Warm        #
        ###############

        warm_users=segment_pd.loc[lambda x: x.segment=='warm']
        num_warm=len(warm_users)
        if num_warm:
            recommendation_warm_users=\
                self._association_rule.recommend_batch(user_ids=warm_users.user_id.values,n_recommends=n_recommends)\
                .assign(segment='warm')

        #############
        # Old       #
        #############
        
        old_users=segment_pd.loc[lambda x:x.segment=='old']
        num_old=len(old_users)
        if num_old:
            recommendation_old_users=\
                self._cf_model.recommend_batch(old_users.user_id.values,n_recommends=n_recommends)\
                .assign(segment='old')

        recommendation=pd.DataFrame()
        if num_new:
            recommendation=pd.concat([recommendation,recommendation_new_users])
        
        if num_warm:
            recommendation=pd.concat([recommendation,recommendation_warm_users])
        
        if num_old:
            recommendation=pd.concat([recommendation,recommendation_old_users])
    
        return recommendation











