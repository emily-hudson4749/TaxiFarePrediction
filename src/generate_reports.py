import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd 
import geopandas as gpd
from shapely.geometry import Point

def fare_distribution_report(df_final):
    # delete outliers
    Q1 = df_final['fare_amount'].quantile(0.25)
    Q3 = df_final['fare_amount'].quantile(0.75)
    IQR = Q3 - Q1
    df_final = df_final[(df_final['fare_amount'] >= Q1 - 1.5 * IQR) & (df_final['fare_amount'] <= Q3 + 1.5 * IQR)]

    # Distribution of Fares
    plt.figure(figsize=(10, 6))
    plt.hist(df_final['fare_amount'], bins=15, color='skyblue', edgecolor='black')
    plt.title('Distribution of Fare Amounts')
    plt.xlabel('Fare Amount ($)')
    plt.ylabel('Frequency')
    plt.savefig('reports/fare_distribution.png')
    plt.close()

def normalized_fare_distribution_report(df_final):
    df_final['log_fare_amount'] = np.log(df_final['fare_amount'])

    # delete outliers
    Q1 = df_final['log_fare_amount'].quantile(0.25)
    Q3 = df_final['log_fare_amount'].quantile(0.75)
    IQR = Q3 - Q1
    df_final = df_final[(df_final['log_fare_amount'] >= Q1 - 1.5 * IQR) & (df_final['log_fare_amount'] <= Q3 + 1.5 * IQR)]

    # Distribution of Fares
    plt.figure(figsize=(10, 6))
    plt.hist(df_final['log_fare_amount'], bins=15, color='coral', edgecolor='black')
    plt.title('Distribution of Fare Amounts')
    plt.xlabel('Fare Amount ($)')
    plt.ylabel('Frequency')
    plt.savefig('reports/log_fare_distribution.png')
    plt.close()

def trip_distribution_report(df_final):
    # delete outliers
    Q1 = df_final['trip_distance'].quantile(0.25)
    Q3 = df_final['trip_distance'].quantile(0.75)
    IQR = Q3 - Q1
    df_final = df_final[(df_final['trip_distance'] >= Q1 - 1.5 * IQR) & (df_final['trip_distance'] <= Q3 + 1.5 * IQR)]

    # Distribution of Distance
    plt.figure(figsize=(10, 6))
    plt.hist(df_final['trip_distance'], bins=15, color='skyblue', edgecolor='black')
    plt.title('Distribution of Trip Distance')
    plt.xlabel('Trip Distance (Miles)')
    plt.ylabel('Frequency')
    plt.savefig('reports/trip_distance_distribution.png')
    plt.close()

def normalized_trip_distribution_report(df_final):
    df_final['log_trip_distance'] = np.log(df_final['trip_distance'])

    # delete outliers
    Q1 = df_final['log_trip_distance'].quantile(0.25)
    Q3 = df_final['log_trip_distance'].quantile(0.75)
    IQR = Q3 - Q1
    df_final = df_final[(df_final['log_trip_distance'] >= Q1 - 1.5 * IQR) & (df_final['log_trip_distance'] <= Q3 + 1.5 * IQR)]


    # Distribution of Distance
    plt.figure(figsize=(10, 6))
    plt.hist(df_final['log_trip_distance'], bins=15, color='coral', edgecolor='black')
    plt.title('Distribution of Trip Distance')
    plt.xlabel('Trip Distance (Miles)')
    plt.ylabel('Frequency')
    plt.savefig('reports/log_trip_distance_distribution.png')
    plt.close()

def fare_across_week_report(combined_df):
    combined_df['fare_per_mile'] = combined_df['fare_amount'] / combined_df['trip_distance'].replace(0, pd.NA)
    combined_df[['pickup_date', 'pickup_time']] = combined_df['tpep_pickup_datetime'].astype(str).str.split(' ', expand = True)
    combined_df['pickup_date'] = pd.to_datetime(combined_df['pickup_date'])
    combined_df['day_of_week'] = combined_df['pickup_date'].dt.day_name()
    # Shorten day names
    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_map = {'Monday': 'Mon', 'Tuesday': 'Tue', 'Wednesday': 'Wed', 'Thursday': 'Thu','Friday': 'Fri', 'Saturday': 'Sat', 'Sunday': 'Sun'}
    combined_df['day_abbr'] = combined_df['day_of_week'].map(day_map)

    heatmap_data = combined_df.pivot_table(index='day_abbr', columns='pickup_hour', values='fare_per_mile', aggfunc='median'
    ).reindex(day_order)

    plt.figure(figsize=(18, 5))

    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, fmt=".2f", cbar_kws={'label': 'Median fare / mile ($)'})

    plt.title('Fare / Distance Heatmap: Hour by Day of Week', fontsize=15, fontweight='bold')
    plt.xlabel('Pickup Hour')
    plt.ylabel('Day of Week')

    plt.xticks(rotation=45)
    plt.savefig('reports/fare_by_day_of_week.png')
    plt.close()

def fare_by_neighborhood_report(df_analysis):
    df_analysis = df_analysis[df_analysis['trip_distance'] > 0]

    df_analysis['price_per_mile'] = df_analysis['fare_amount'] / df_analysis['trip_distance']


    df_analysis['route'] = df_analysis['PU_Borough'] + " to " + df_analysis['DO_Borough']

    # Rush Hour (7-10 AM and 4-7 PM)
    def is_rush_hour(hour):
        return 1 if (7 <= hour <= 10) or (16 <= hour <= 19) else 0

    df_analysis['is_rush_hour'] = df_analysis['pickup_hour'].apply(is_rush_hour)

    top_routes = df_analysis['route'].value_counts().nlargest(10).index
    df_top = df_analysis[df_analysis['route'].isin(top_routes)]

    route_analysis = df_top.groupby(['route', 'is_rush_hour'])['price_per_mile'].mean().unstack()

    plt.figure(figsize=(12, 8))
    ax = route_analysis.plot(kind='barh', color=['skyblue', 'salmon'], figsize=(12, 8))

    plt.title('Average Price per Mile: Normal vs. Rush Hour by Route', fontsize=14, pad=20)
    plt.xlabel('Average Fare per Mile ($)', fontsize=12)
    plt.ylabel('Borough Pair', fontsize=12)

    plt.legend(['Off-Peak', 'Rush Hour'],
            title='Time of Day',
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.)
    plt.savefig('reports/fare_by_neighborhood.png')
    plt.close()

def fare_by_weather_report(df_final):
    df_final['temp_type'] = np.where(df_final['temp_max'] <= 32, 'Cold', 'Warm')
    df_final['precip_type'] = np.where(df_final['precipitation'] > 0.5, 'Rainy', 'Dry')
    df_final['weather_state'] = df_final['temp_type'] + " & " + df_final['precip_type']


    df_plot = df_final[df_final['trip_distance'] > 0.1].copy()
    df_plot['fare_per_mile'] = df_plot['fare_amount'] / df_plot['trip_distance']


    plt.figure(figsize=(12, 7))

    sns.boxplot(
        data=df_plot,
        x='weather_state',
        y='fare_per_mile',
        palette='coolwarm',
        showfliers=False  
    )

    plt.title('Fare Efficiency Distribution by Weather Condition', fontsize=15)
    plt.ylabel('Fare Amount per Mile ($)', fontsize=12)
    plt.xlabel('Weather State (Temp & Precipitation)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('reports/fare_by_weather.png')
    plt.close()

def geospatial_fare_report(df_final):
    geometry = [
    Point(xy) for xy in zip(df_final['LocationID_x'], df_final['LocationID_y'])
]

    gdf = gpd.GeoDataFrame(df_final, geometry=geometry)

    
    gdf.plot(
        column='fare_amount',   
        cmap='viridis',         
        markersize=2,
        alpha=0.6,
        legend=True
    )

    plt.title("Pickup Locations Colored by Fare")
    plt.savefig('reports/geospatial_fare_report.png')
    plt.close()