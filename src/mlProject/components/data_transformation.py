import joblib
import pandas as pd
from mlProject import logger
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import numpy as np
from mlProject import logger
from sklearn.model_selection import train_test_split
import os 
from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    
    def __init__(self,config: DataTransformationConfig):
        self.config = config
         
         

    def dropUnnecessaryColumns(self,data,columnNameList): 
        data= data = pd.read_csv(self.config.data_path)
        data = data.drop(columnNameList,axis=1)
        return data
   
    
    

    def replaceInvalidValuesWithNull(self,data):
        for column in data.columns:
            count = data[column][data[column] == '?'].count()
            if count != 0:
                data[column] = data[column].replace('?', np.nan)
        return data
    
    
    def encodeCategoricalValues(self,data):
         
    # We can map the categorical values like below:
        data['sex'] = data['sex'].map({'F': 0, 'M': 1})

     # except for 'Sex' column all the other columns with two categorical data have same value 'f' and 't'.
     # so instead of mapping indvidually, let's do a smarter work
        for column in data.columns:
            if len(data[column].unique()) == 2:
                data[column] = data[column].map({'f': 0, 't': 1})

     # this will map all the rest of the columns as we require. Now there are handful of column left with more than 2 categories.
     # we will use get_dummies with that.
        data = pd.get_dummies(data,columns=['referral_source'])

        encode = LabelEncoder().fit(data['Class'])

        data['Class'] = encode.transform(data['Class'])


    # we will save the encoder as pickle to use when we do the prediction. We will need to decode the predcited values
    # back to original
        #with open('EncoderPickle/enc.pickle', 'wb') as file:
            #joblib.dump(encode, file)
        joblib.dump(encode, os.path.join(self.config.root_dir, self.config.encoder_name))

        return data
    

    
    def impute_missing_values(self,data):
        imputer=KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan)
        new_array=imputer.fit_transform(data)
        data=pd.DataFrame(data=np.round(new_array), columns=data.columns)
        
        return data
    

    def train_test_spliting(self,data):
        #data = pd.read_csv(self.config.data_path)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
    