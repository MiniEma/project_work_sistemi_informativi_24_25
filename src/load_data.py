import pandas as pd 
import sys 
import os 
import logging
import src.config as config
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") # log set up

def load_data() :
    #Load the data from the CSV file and return a DataFrame."""
    data = pd.read_excel(os.path.join(config.DATA_PATH,'Housing_data.xlsx'))
    logging.info("Data loaded successfully.")
    return data