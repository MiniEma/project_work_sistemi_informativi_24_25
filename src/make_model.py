import pandas as pd
import os 
import sys
from src import config
from src import load_data    
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from xgboost import XGBRegressor
import pickle 
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s") # log set up

def train_model(data):
    #split the data into features and target variable
    
    mod1 = data[["house_age","distance_to_nearest_MRT_station", "number_of_convenience_stores"]]
    mod2 = data[["latitude","longitude"]]
    
    y = data['house_price']
    # Split the data into training and testing sets
    mod1_train, mod1_test, y_train, y_test = train_test_split(mod1, y, test_size=0.2, random_state=33)
    # Initialize the model
    lf1= LinearRegression()
    rf1 = RandomForestRegressor()
    xb1 = XGBRegressor()
    # Train the model
    rf1.fit(mod1_train, y_train)
    lf1.fit(mod1_train, y_train)
    xb1.fit(mod1_train, y_train)
    # Save the model to a file
    with open(os.path.join(config.MODEL_PATH, 'rf_model_hdn.pkl'), 'wb') as f:
        pickle.dump(rf1, f)
    # Save the model to a file
    with open(os.path.join(config.MODEL_PATH, 'lf_model_hdn.pkl'), 'wb') as f:
        pickle.dump(lf1, f)
    # Save the model to a file
    with open(os.path.join(config.MODEL_PATH, 'xb_model_hdn.pkl'), 'wb') as f:
        pickle.dump(xb1, f)
    
    
    # Make predictions on the test set
    rf_predictions = rf1.predict(mod1_test)
    lf_predictions = lf1.predict(mod1_test)
    xb_predictions = xb1.predict(mod1_test)
    # Evaluate the model
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_r2 = r2_score(y_test, rf_predictions)
    lf_mse = mean_squared_error(y_test, lf_predictions)
    lf_mae = mean_absolute_error(y_test, lf_predictions)
    lf_r2 = r2_score(y_test, lf_predictions)
    xb_mse = mean_squared_error(y_test, xb_predictions)
    xb_mae = mean_absolute_error(y_test, xb_predictions)
    xb_r2 = r2_score(y_test, xb_predictions)
    
    # Save the evaluation metrics to a file
    with open(os.path.join(config.MODEL_PATH, 'evaluation_metrics.txt'), 'w') as f:
        f.write("Random Forest Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {rf_mse}\n")
        f.write(f"Mean Absolute Error (MAE): {rf_mae}\n")
        f.write(f"R-squared (R2): {rf_r2}\n\n")
        f.write("Linear Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {lf_mse}\n")
        f.write(f"Mean Absolute Error (MAE): {lf_mae}\n")
        f.write(f"R-squared (R2): {lf_r2}\n\n")
        f.write("XGBoost Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {xb_mse}\n")
        f.write(f"Mean Absolute Error (MAE): {xb_mae}\n")
        f.write(f"R-squared (R2): {xb_r2}\n\n")
    

    
    # initiazling mod2 model
    mod2_train, mod2_test, y_train, y_test = train_test_split(mod2, y, test_size=0.2, random_state=33)
    # Initialize the model  
    lf2 = LinearRegression()
    rf2 = RandomForestRegressor()
    xb2 = XGBRegressor()
    # Train the model
    rf2.fit(mod2_train, y_train)
    lf2.fit(mod2_train, y_train)
    xb2.fit(mod2_train, y_train)
    
    # Save the models to a file
    with open(os.path.join(config.MODEL_PATH, 'rf_model_ll.pkl'), 'wb') as f:
        pickle.dump(rf2, f)

    with open(os.path.join(config.MODEL_PATH, 'lf_model_ll.pkl'), 'wb') as f:
        pickle.dump(lf2, f)
    
    with open(os.path.join(config.MODEL_PATH, 'xb_model_ll.pkl'), 'wb') as f:
        pickle.dump(xb2, f)

    # Make predictions on the test set
    rf_predictions = rf2.predict(mod2_test)
    lf_predictions = lf2.predict(mod2_test)
    xb_predictions = xb2.predict(mod2_test)
    # Evaluate the model
    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_r2 = r2_score(y_test, rf_predictions)
    lf_mse = mean_squared_error(y_test, lf_predictions)
    lf_mae = mean_absolute_error(y_test, lf_predictions)
    lf_r2 = r2_score(y_test, lf_predictions)
    xb_mse = mean_squared_error(y_test, xb_predictions)
    xb_mae = mean_absolute_error(y_test, xb_predictions)
    xb_r2 = r2_score(y_test, xb_predictions)
    
    
    # Save the evaluation metrics to a file
    with open(os.path.join(config.MODEL_PATH, 'evaluation_metrics.txt'), 'a') as f:
        f.write("\nRandom Forest Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {rf_mse}\n")
        f.write(f"Mean Absolute Error (MAE): {rf_mae}\n")
        f.write(f"R-squared (R2): {rf_r2}\n\n")
        f.write("Linear Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {lf_mse}\n")
        f.write(f"Mean Absolute Error (MAE): {lf_mae}\n")
        f.write(f"R-squared (R2): {lf_r2}\n\n")
        f.write("XGBoost Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {xb_mse}\n")
        f.write(f"Mean Absolute Error (MAE): {xb_mae}\n")
        f.write(f"R-squared (R2): {xb_r2}\n\n")

