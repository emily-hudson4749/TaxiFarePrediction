from sklearn.metrics import mean_squared_error
import numpy as np
from train_models import model_cols
    

def evaluate_model(model, test_df):
   
    y_pred = model.predict(test_df[model_cols])
    rmse = np.sqrt(mean_squared_error(test_df['log_fare_amount'], y_pred))
    
    return rmse