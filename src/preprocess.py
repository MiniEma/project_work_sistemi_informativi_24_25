import pandas as pd   
import sys
import os
from src import config
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") # log set up
def preprocess_data(data):
    logging.info("Preprocessing data...")
    
    # Drop unnecessary columns
    data = data.drop(columns=['No', 'X1 transaction date']) 

    # Rename columns for clarity
    data.columns = ['house_age', 'distance_to_nearest_MRT_station', 
                    'number_of_convenience_stores', 'latitude', 'longitude', 'house_price']
    data.dropna(inplace=True)  # Drop rows with missing values
    data.drop_duplicates(inplace=True)  # Drop duplicate rows   
    data = data.reset_index(drop=True)  # Reset index after dropping rows
    #saeve the preprocessed data to a new file
    data.to_csv(os.path.join(config.DATA_PATH, 'preprocessed_data.csv'), index=False)
    logging.info("Preprocessing completed and data saved to preprocessed_data.csv.")
    return data

