from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_transformation import DataTransformation
from mlProject import logger
from pathlib import Path



STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                
                data_transformation_config = config.get_data_transformation_config()
    
                data_trnsformation = DataTransformation( config = data_transformation_config)
     
                data = data_trnsformation.get_data()
    
                data = data_trnsformation.dropUnnecessaryColumns(data,['TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured','TBG_measured','TBG','TSH'])
     
                data = data_trnsformation.replaceInvalidValuesWithNull(data)
 
                data = data_trnsformation.encodeCategoricalValues(data)
    
                data= data_trnsformation.impute_missing_values(data)
    
                X, Y = data_trnsformation.separate_label_feature(data, label_column_name='Class')
                
                X_sampled, Y_sampled = data_trnsformation.handleImbalanceDataset(X,Y)

                data = data_trnsformation.train_test_spliting(X_sampled,Y_sampled)
 
            else:
                raise Exception("You data schema is not valid")

        except Exception as e:
            print(e)