from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path



@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    encoder_name: str


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_x_data_path: Path
    test_x_data_path: Path
    train_y_data_path: Path
    test_y_data_path: Path
    model_name: str
    criterion: str
    splitter: str
    max_depth: str
    min_samples_split: int
    min_samples_leaf: int
    min_weight_fraction_leaf: float
    max_features: str
    random_state: int
    max_leaf_nodes: str
    min_impurity_decrease: float
    class_weight: str
    ccp_alpha: float
    target_column: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_x_data_path: Path
    test_y_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str