import os
import sys
sys.path.append(os.path.abspath('..')) 

import logging
import src
from src import config
from src.load_data import load_data
from src.preprocess import preprocess_data
from src.make_model import train_model


# Set up logging
logging.basicConfig(filename='../log/pipeline.log', level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")




def main():
    logging.info("Starting  Pipeline...")

    # Step 1: Load raw data
    logging.info("Loading raw data...")
    data = load_data()

    # Step 2: Preprocess  data
    logging.info("Preprocessing data...")
    datap = preprocess_data(data)

    # Step 3: Train model
    logging.info("Training the model...")
    train_model(datap)

if __name__ == "__main__":
    main()
