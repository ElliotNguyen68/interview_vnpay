from abc import ABC,abstractmethod
from typing import List,Union,Any

import pandas as pd

class BaseRecommender(ABC):
    def __init__(self) -> None:
        pass

    def fit(self,data:pd.DataFrame):
        self._validate_data(data)
        self.data=data 

    @abstractmethod
    def _validate_data(self, data: pd.DataFrame):
        """_summary_

        Args:
            data (pd.DataFrame): _description_
        """

    def retrain(self,new_data:Union[pd.DataFrame,Any]):
        """_summary_

        Args:
            new_data (pd.DataFrame): _description_
        """
        self.fit(new_data)

    @abstractmethod
    def recommend_1(self,user_id:int):
        """_summary_

        Args:
            user_id (int): _description_
        """

    
    def _validate_data(self,data:pd.DataFrame):
        """_summary_

        Args:
            data (pd.DataFrame): _description_
        """
    

    

        
    