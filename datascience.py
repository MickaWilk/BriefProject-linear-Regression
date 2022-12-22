from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import numpy as np

def create_pipeline(model=LinearRegression()):
    num_pipeline = Pipeline(
        [
        ('imputer', SimpleImputer(strategy="median")),
        ('minmax_scaler', MinMaxScaler())
        ]
    )
    cat_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')
    preprocessor = ColumnTransformer([
        ('num_transformer', num_pipeline, make_column_selector(dtype_include = np.number)),
        ('cat_transformer', cat_transformer, make_column_selector(dtype_exclude = np.number))
        ])
    pipeline_workflow = make_pipeline(preprocessor, model)
    return pipeline_workflow
