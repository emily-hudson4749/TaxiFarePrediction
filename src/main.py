from data_prep import *
from generate_reports import *
from train_models import *
from evaluate import *

def main():
    
    df = download_trip_info()

    taxi_data = clean_taxi_data(df)
    
    weather = format_weather_data(pd.read_csv("data/raw/nyc_weather.csv"))
    gas_prices = format_gas_data(pd.read_csv("data/raw/nyc_gas_prices.csv"))
    neighborhoods = pd.read_csv("data/raw/taxi_zone_lookup.csv")

    df_final = merge_datasets(taxi_data, weather, neighborhoods, gas_prices)

    ##generate reports
    fare_distribution_report(df_final)
    normalized_fare_distribution_report(df_final)
    trip_distribution_report(df_final)
    normalized_trip_distribution_report(df_final)
    fare_across_week_report(df_final)
    fare_by_weather_report(df_final)
    ##geospatial_fare_report(df_final) need to work on this 
    
    train_df, val_df, test_df = create_test_split(df_final)

    train_df = add_neighborhood_features(train_df, neighborhoods)
    val_df = add_neighborhood_features(val_df, neighborhoods)
    test_df = add_neighborhood_features(test_df, neighborhoods)

    combined = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    fare_by_neighborhood_report(combined)

    train_df, val_df, test_df = encode_data(train_df, val_df, test_df)

    rf_model = train_random_forest(train_df, val_df)

    xgb_model = train_xgboost(train_df, val_df)

    rf_rmse = evaluate_model(rf_model, test_df)
    xgb_rmse = evaluate_model(xgb_model, test_df)

    print(f"Random Forest RMSE: {rf_rmse}")
    print(f"XGBoost RMSE: {xgb_rmse}")


if __name__ == "__main__":
    main()


