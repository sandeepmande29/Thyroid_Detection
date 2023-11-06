import pandas as pd
import os
from mlProject import logger
from sklearn.tree import DecisionTreeClassifier
import joblib
from mlProject.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_x = pd.read_csv(self.config.train_x_data_path)
        train_y = pd.read_csv(self.config.train_y_data_path)
 

        dt = DecisionTreeClassifier(criterion= self.config.criterion,splitter= self.config.splitter,random_state = self.config.random_state,ccp_alpha=self.config.ccp_alpha)
        dt.fit(train_x, train_y)

        joblib.dump(dt, os.path.join(self.config.root_dir, self.config.model_name))