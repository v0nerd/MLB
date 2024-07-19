import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import pandas as pd
import openpyxl
from matplotlib import pyplot as plt

from keras._tf_keras.keras.models import load_model, Sequential
from keras._tf_keras.keras.layers import Dense, Dropout, InputLayer
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras._tf_keras.keras.optimizers as optimizers
from sklearn.model_selection import train_test_split

def load_or_run_model(scalers:dict, X_scaled:np.ndarray, y_scaled:np.ndarray):

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    file_path = 'data_and_models/basic_model_comb.keras'

    if not os.path.exists(file_path):
        
        ## Model is basic version without any callibration including no change to hyperparameters ##

        model = Sequential([
            InputLayer(shape = (X_scaled.shape[1],)),
            Dense(units=2, activation = 'relu')
        ])  

        opt = optimizers.Adam()
        model.compile(optimizer = opt, loss='mean_squared_error')
        es = EarlyStopping(monitor = 'loss', mode = 'min', patience = 6)
        mcp_save = ModelCheckpoint(file_path, save_best_only=True, monitor='loss', mode='min')
        model.fit(X_train, y_train, epochs=150, batch_size=32, callbacks = [es, mcp_save]) 

        predicted_scores_test = model.predict(X_test)

        # Rescale back to original range    
        home_predicted_scores = scalers['HT_SC'].inverse_transform(predicted_scores_test[:, 0].reshape(-1, 1))    
        away_predicted_scores = scalers['AT_SC'].inverse_transform(predicted_scores_test[:, 1].reshape(-1, 1))
        y_test_home = np.round(scalers['HT_SC'].inverse_transform(y_test[:, 0].reshape(-1, 1)))
        y_test_away = np.round(scalers['AT_SC'].inverse_transform(y_test[:, 1].reshape(-1, 1)))
        
        ### Evaluate ### 
        correct_score_counter = 0
        correct_outcome_counter = 0
        total_fixtures = len(home_predicted_scores)
    
        home_predicted_scores_rounded = []
        away_predicted_scores_rounded = []
        
        for index in range(0, len(home_predicted_scores), 1):

            home_pred = round(home_predicted_scores[index][0])
            away_pred = round(away_predicted_scores[index][0]) 

            if home_pred == away_pred and home_predicted_scores[index] > away_predicted_scores[index]:
                away_pred -= 1
            elif home_pred == away_pred and home_predicted_scores[index] < away_predicted_scores[index]:
                home_pred -= 1

            home_predicted_scores_rounded.append(home_pred)
            away_predicted_scores_rounded.append(away_pred)

            print('H vs A predicted...  ', home_pred, ' : ', away_pred)
            print('H vs A actual...  ', y_test_home[index][0], ' : ', y_test_away[index][0])
            print('*****************************************************************')
            if home_pred == y_test_home[index] and away_pred == y_test_away[index]:
                correct_score_counter += 1
                correct_outcome_counter += 1
            elif home_pred > away_pred and y_test_home[index] > y_test_away[index]:
                correct_outcome_counter += 1
            elif home_pred < away_pred and y_test_home[index] < y_test_away[index]:
                correct_outcome_counter += 1
                
        correct_score_pct = (correct_score_counter / total_fixtures) * 100
        correct_outcome_pct = (correct_outcome_counter / total_fixtures) * 100
        print('Successful score prediction pct = ' + str(round(correct_score_pct, 2)) + ', Successful outcome prediction pct = ' + str(round(correct_outcome_pct, 2)))

        home_predicted_scores_rounded = np.array(home_predicted_scores_rounded)
        away_predicted_scores_rounded = np.array(away_predicted_scores_rounded)

        ## Home Scores ##
        home_mse_test = mean_squared_error(y_test_home, home_predicted_scores_rounded)
        home_MAE_test = mean_absolute_error(y_test_home, home_predicted_scores_rounded)
        home_R2val_test = r2_score(y_test_home, home_predicted_scores_rounded)
        ## Away Scores
        away_mse_test = mean_squared_error(y_test_away, away_predicted_scores_rounded)
        away_MAE_test = mean_absolute_error(y_test_away, away_predicted_scores_rounded)
        away_R2val_test = r2_score(y_test_away, away_predicted_scores_rounded)
        
        print('RMSE = ' + str(home_mse_test) + ',' + str(away_mse_test) + ' MAE = ' + str(home_MAE_test) + ',' + str(away_MAE_test) + ' R2 val = ' + str(home_R2val_test) + ',' + str(away_R2val_test))
    else:
        model = load_model(file_path)
    
    return model