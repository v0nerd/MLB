from retrieve_data import get_data, scale_data, prep_pred_input
from model import load_or_run_model
from retrieve_data import update_current_team_database
import numpy as np

def activate():

    data = get_data()

    scalers, X_scaled, y_scaled = scale_data(data)
    
    model = load_or_run_model(scalers, X_scaled, y_scaled)

    ### This is where the new predictions will get read in ###
    list_of_fixtures = [
        {'DATE': '2024-07-12', 'HT': 'Los Angeles Dodgers', 'AT' : 'Detroit Tigers'}, 
        {'DATE': '2024-07-12', 'HT': 'Oakland Athletics', 'AT' : 'Philadelphia Phillies'}
        ]
    
    for fixture in list_of_fixtures:

        pred_input, hist_score = prep_pred_input(fixture, scalers)

        predicted_outcome = model.predict(pred_input)

        ## Baseball always have a winner, following code makes sure there is a score difference ##

        home_pred_unrounded = scalers['HT_SC'].inverse_transform(predicted_outcome[:, 0].reshape(-1, 1))[0][0]
        away_pred_unrounded = scalers['AT_SC'].inverse_transform(predicted_outcome[:, 1].reshape(-1, 1))[0][0]

        home_pred = round(home_pred_unrounded)
        away_pred = round(away_pred_unrounded) 

        if home_pred == away_pred and home_pred_unrounded > away_pred_unrounded:
            away_pred -= 1
        elif home_pred == away_pred and home_pred_unrounded < away_pred_unrounded:
            home_pred -= 1

        print(fixture['HT'], ':', fixture['AT'], ' predicted score... ', home_pred, ':', away_pred, 'actual score :', hist_score)

activate()

def update_current_data():

    update_current_team_database()

    print('Ready to create new predictions.')

