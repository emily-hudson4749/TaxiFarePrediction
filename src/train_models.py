from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from data_prep import encode_data


model_cols = [
    'LocationID_x', 'LocationID_y', 'PU_Borough', 'DO_Borough',
    'usd_per_gallon', 'pickup_hour', 'day_of_week_Friday',	'day_of_week_Monday',	'day_of_week_Saturday',
    'day_of_week_Sunday',	'day_of_week_Thursday', 'day_of_week_Tuesday',	'day_of_week_Wednesday',
    'log_trip_distance', 'Airport_fee', 'is_airport', 'inter_borough', 'is_weekend'
]


def train_random_forest(train_df,  val_df):

    TARGET_RMSE = 10000
    
    model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
    )

    train_encoded, val_encoded, x = encode_data(train_df, val_df, val_df) ##not using test here 

    model.fit(train_encoded[model_cols], train_df['log_fare_amount'])


    val_preds = model.predict(val_encoded[model_cols])
    val_rmse = np.sqrt(mean_squared_error(val_df['log_fare_amount'], val_preds))

    if val_rmse <= TARGET_RMSE:
        best_model = model

    else:

        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [10, 20, 30],
            "min_samples_leaf": [1, 5, 10],
            "max_features": ["sqrt", 0.5, 0.8]
        }

        grid = GridSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid,
            scoring="root_mean_squared_error",
            cv=3,
            n_jobs=-1
        )

        grid.fit(train_encoded[model_cols], train_df['log_fare_amount'])

        best_model = grid.best_estimator_    


    joblib.dump(best_model, "models/random_forest_model.pkl")
    
    return best_model



def train_xgboost(train_df,  val_df):

    TARGET_RMSE = 10000
    
    model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=10, ##max depth of a tree 
    subsample=0.7, ##subsample of training data, helps with overfitting 
    reg_lambda=5, ##regularization 
    min_child_weight=3, ##min sum of weight needed in a child 
    random_state=42
    )

    train_encoded, val_encoded, x = encode_data(train_df, val_df, val_df) ##not using test here 

    model.fit(train_encoded[model_cols], train_df['log_fare_amount'])


    val_preds = model.predict(val_encoded[model_cols])
    val_rmse = np.sqrt(mean_squared_error(val_df['log_fare_amount'], val_preds))

    if val_rmse <= TARGET_RMSE:
        best_model = model

    else:

        param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 5],
        "gamma": [0, 0.1]
        }

        grid = GridSearchCV(
            XGBRegressor(random_state=42),
            param_grid,
            scoring="root_mean_squared_error",
            cv=3,
            n_jobs=-1
        )

        grid.fit(train_encoded[model_cols], train_df['log_fare_amount'])

        best_model = grid.best_estimator_    


    joblib.dump(best_model, "models/xgboost_model.pkl")
    
    return best_model