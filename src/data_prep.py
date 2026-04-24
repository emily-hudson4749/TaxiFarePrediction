import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split  
import category_encoders as ce
  

def download_trip_info():
    ##downloading kaggle dataset  
    df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "aryanpatel212/cleaned-nyc-taxi-trip-data-2025-sample",
    "NYC_Taxi_Cleaned_Analysis_Ready.csv"
    )
    df = df.sample(n=1000000, random_state = 10)
    return df

def clean_taxi_data(taxi_trips):
    
    ##creating pickup and dropoff date and time columns
    taxi_trips[['pickup_date', 'pickup_time']] = taxi_trips['tpep_pickup_datetime'].astype(str).str.split(' ', expand = True)
    taxi_trips[['dropoff_date', 'dropoff_time']] = taxi_trips['tpep_dropoff_datetime'].astype(str).str.split(' ', expand = True)
    
    ##formatting 
    taxi_trips['pickup_date'] = pd.to_datetime(taxi_trips['pickup_date'])
    taxi_trips['dropoff_date'] = pd.to_datetime(taxi_trips['dropoff_date'])

    ##creating day of week column and one hot encoding it
    taxi_trips['day_of_week'] = taxi_trips['pickup_date'].dt.day_name()
    taxi_trips = pd.get_dummies(taxi_trips, columns = ['day_of_week'])
    
    return taxi_trips

def format_weather_data(weather):
    ##formatting to date time 
    weather['date'] = pd.to_datetime(weather['date'])
    return weather

def format_gas_data(gas_prices):
    ##formatting to date time 
    gas_prices['Week of'] = pd.to_datetime(gas_prices['Week of'])
    return gas_prices

def merge_datasets(taxi_trips, weather, neighborhoods, gas_prices):
    combined_df = taxi_trips.merge(weather, left_on="pickup_date", right_on="date", how="left")

    # Neighborhood merges
    combined_df = combined_df.merge(neighborhoods, left_on='PULocationID', right_on='LocationID', how = 'left').rename(columns= {'Borough':'pickup_borough', 'Zone': 'pickup_zone'})

    combined_df = combined_df.merge(neighborhoods, left_on='DOLocationID', right_on='LocationID', how='left').rename(columns={'Borough': 'dropoff_borough', 'Zone': 'dropoff_zone'})

    # Gas merge
    combined_df['week_start'] = combined_df['pickup_date'] - pd.to_timedelta(combined_df['pickup_date'].dt.weekday, unit = 'D')
    combined_df = combined_df.merge(gas_prices, left_on="week_start", right_on="Week of", how = 'left')
   
    ##selecting final features 
    final_cols = ['LocationID_x', 'LocationID_y', 'temp_max','usd_per_gallon', 'fare_amount',  'pickup_hour', 'trip_distance', 'Airport_fee', 'precipitation',
              'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'is_weekend', 'day_of_week_Friday',	'day_of_week_Monday',	'day_of_week_Saturday',
              'day_of_week_Sunday',	'day_of_week_Thursday', 'day_of_week_Tuesday',	'day_of_week_Wednesday'
]

    df_final = combined_df[final_cols]

    return df_final

def create_test_split(df_final):

    train_df, test_df = train_test_split(df_final, test_size=0.2, random_state=17)
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=38)

    return  train_df, val_df, test_df


def add_neighborhood_features(df, neighborhoods):
    df = df.merge(neighborhoods, left_on='LocationID_x', right_on='LocationID', how='left')
    df = df.rename(columns={'Borough': 'PU_Borough'})
    df = df.drop(columns=['LocationID'])
    df = df.merge(neighborhoods, left_on='LocationID_y', right_on='LocationID', how='left')
    df = df.rename(columns={'Borough': 'DO_Borough'})
    df = df.drop(columns=['LocationID'])

    # Feature Engineering
    df['is_airport'] = df['LocationID_x'].isin([1, 132, 138]) | df['LocationID_y'].isin([1, 132, 138])
    df['inter_borough'] = (df['PU_Borough'] != df['DO_Borough']).astype(int)

    return df

def encode_data(train_df, val_df, test_df):
    target_cols = ['LocationID_x', 'LocationID_y', 'PU_Borough', 'DO_Borough']
    encoder = ce.TargetEncoder(cols=target_cols, smoothing=10)

    train_encoded = encoder.fit_transform(train_df, train_df['log_fare_amount'])
    val_encoded = encoder.transform(val_df)
    test_encoded = encoder.transform(test_df)

    return train_encoded, val_encoded, test_encoded

