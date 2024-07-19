import pandas as pd
from datetime import datetime
import requests
import matplotlib.pyplot as plt
import numpy as np

def build_db():

    ## Going back to 2000, retrieving data from each year, separating every fixture into a separate entry and then creating a csv file ##

    current_year = datetime.now().year 
    years_back = 23 
    years_of_interest = [int(year) for year in range(current_year - years_back, current_year + 1) if year != 2020]

    years_fixture_df = pd.DataFrame()
    for year in years_of_interest:

        year_str = str(year)
        response = requests.get(f'https://www.thesportsdb.com/api/v1/json/60130162/eventsseason.php?id=4424&s={year_str}')

        year_details = response.json()
        individual_games = year_details['events']
        fixtures_df = pd.DataFrame(individual_games)

        if not len(years_fixture_df):
            years_fixture_df = fixtures_df
        else:
            years_fixture_df = pd.concat([fixtures_df, years_fixture_df],  ignore_index = True)

    years_fixture_df.to_csv('data_and_models/mlb_fixture_details.csv')

    ## Get all fixtures available from screener
    #response = requests.get('https://www.thesportsdb.com/api/v1/json/60130162/eventsseason.php?id=4424&s=2023')
    # print(response.text)
    #response = requests.get('https://www.thesportsdb.com/api/v1/json/60130162/lookuplineup.php?id=549877')
    # response = requests.get('https://www.thesportsdb.com/api/v2/json/60130162/lookup/event_lineup/480008')
    # print(response.text)

    ## baseball ID = 105
    ## idleague = 4424
    

    ## All Events in a specific league by season
    ## Events on a specific day

def extract_input_info_from_db():

    initial_db = pd.read_csv('data_and_models/mlb_fixture_details.csv')

    # make sure sorted by dateEvent
    initial_db = initial_db.sort_values(by = ['dateEvent'])

    # filtering out any rows with no scores (and no hits)
    initial_db = initial_db[
        initial_db['intHomeScore'].notna() & (initial_db['intHomeScore'] != '') &
        initial_db['intAwayScore'].notna() & (initial_db['intAwayScore'] != '')
    ]

    initial_db = initial_db[
        initial_db['strResult'].notna() & (initial_db['strResult'] != '') 
    ]

    # group by strSeason
    season_groups = initial_db.groupby('strSeason')

    # create dictionaries from teamIDs for runs for, runs against, hits, errors
    unique_team_ids = initial_db['idHomeTeam'].unique()

    # retrieve countable stats from row incl. intHomeScore, intAwayScore, idHomeTeam, idAwayTeam, strResult (Hits + Errors), idEvent, 
    row_index = 0
    variables = ['HT_AT_DATE', 'DATE', 'HT', 'AT', 'HT_RF', 'AT_RF', 'HT_RA', 'AT_RA', 'HT_H', 'AT_H', 'HT_SC', 'AT_SC'] # add any new inputs and adjust outputs as neccessary (possibly errors + ELO?)
    fixtures = pd.DataFrame(columns = variables) 
    for season, group in season_groups:

        runs_for_dict = {ID : 0 for ID in unique_team_ids}
        runs_against_dict = {ID : 0 for ID in unique_team_ids}
        hits_dict = {ID : 0 for ID in unique_team_ids}
        #errors_dict = {ID : 0 for ID in unique_team_ids}

        for index, row in group.iterrows():

            fixture_details = {}

            home_id = row['idHomeTeam']
            away_id = row['idAwayTeam']

            if home_id == 142040 or away_id == 142040:
                continue

            home_ABRV = get_ABRV_from_id(home_id)
            away_ABRV = get_ABRV_from_id(away_id)
            date = row['dateEvent']

            # Add all relevant information to dataframe for specific fixture
            fixture_details['HT_AT_DATE'] = home_ABRV + '_' + away_ABRV + '_' + date
            fixture_details['DATE'] = date

            fixture_details['HT'] = get_teamcode_from_id(home_id)
            fixture_details['AT'] = get_teamcode_from_id(away_id)

            fixture_details['HT_RF'] = runs_for_dict[home_id]
            fixture_details['AT_RF'] = runs_for_dict[away_id]

            fixture_details['HT_RA'] = runs_against_dict[home_id]
            fixture_details['AT_RA'] = runs_against_dict[away_id]

            fixture_details['HT_H'] = hits_dict[home_id]
            fixture_details['AT_H'] = hits_dict[away_id]

            home_total_runs = row['intHomeScore']
            away_total_runs = row['intAwayScore']

            fixture_details['HT_SC'] = home_total_runs
            fixture_details['AT_SC'] = away_total_runs

            # Append stats to dictionaries so summarised statistics can be gathered
            
            runs_for_dict[home_id] += row['intHomeScore']
            runs_for_dict[away_id] += row['intAwayScore']

            runs_against_dict[home_id] += row['intAwayScore']
            runs_against_dict[away_id] += row['intHomeScore']

            fixture_result = row['strResult']
            if 'Team Totals' in fixture_result:
                temp = fixture_result.split('Team Totals')
                home_results_table = temp[1].split('\r\n\r')[0]
                away_results_table = temp[2]
                home_table_split = home_results_table.split(' ')
                home_table_split = list(filter(None, home_table_split))
                away_table_split = away_results_table.split(' ')
                away_table_split = list(filter(None, away_table_split))
                home_hits = int(home_table_split[2])
                away_hits = int(away_table_split[2])
            else:
                if 'Hits' not in fixture_result:
                    continue
                
                team_results = fixture_result.split('Hits:') # - Errors:
                home_hits_str = team_results[1].split('- Errors:')[0]
                if any(char.isdigit() for char in home_hits_str):
                    home_hits = int(home_hits_str)
                else:
                    continue
                away_hits_str = team_results[2].split('- Errors:')[0]
                if any(char.isdigit() for char in away_hits_str):
                    away_hits = int(away_hits_str)
                else:
                    continue
            
            hits_dict[home_id] += home_hits
            hits_dict[away_id] += away_hits

            #errors_dict[home_id] += home_errors
            #errors_dict[away_id] += away_errors
        
            fixtures.loc[row_index] = fixture_details
            print('Fixture Added: ', fixture_details['HT_AT_DATE'])
            row_index += 1
        
        # print(runs_for_dict)
        # print(runs_against_dict)
        # print(hits_dict)

        season_data = [runs_for_dict, runs_against_dict, hits_dict]
        labels = [f'RF_{season}', f'RA_{season}', f'H_{season}']
        totals = pd.DataFrame(season_data, index = labels)        
        #totals.to_csv('data_and_models/totals.csv', mode = 'a')

        print('Group Complete ', season)

    #file_path = 'data_and_models/mlb_model_ready_data.csv'       
    #fixtures.to_csv(file_path, index=False)

def combined_db_creation():

    initial_db = pd.read_csv('data_and_models/mlb_fixture_details.csv')
    
    elo_db = pd.read_csv('data_and_models/mlb_elo.csv')

    # make sure sorted by dateEvent
    initial_db = initial_db.sort_values(by = ['dateEvent'])

    # filtering out any rows with no scores (and no hits)
    initial_db = initial_db[
        initial_db['intHomeScore'].notna() & (initial_db['intHomeScore'] != '') &
        initial_db['intAwayScore'].notna() & (initial_db['intAwayScore'] != '')
    ]

    initial_db = initial_db[
        initial_db['strResult'].notna() & (initial_db['strResult'] != '') 
    ]

    print(len(initial_db))

    # group by strSeason
    season_groups = initial_db.groupby('strSeason')

    # create dictionaries from teamIDs for runs for, runs against, hits, errors
    unique_team_ids = initial_db['idHomeTeam'].unique()

    # retrieve countable stats from row incl. intHomeScore, intAwayScore, idHomeTeam, idAwayTeam, strResult (Hits + Errors), idEvent, 
    row_index = 0
    variables = ['HT_AT_DATE', 'DATE', 'HT', 'AT', 'HT_RD', 'AT_RD', 'HT_ELO', 'AT_ELO', 'HT_HPG', 'AT_HPG', 'HT_PREV_SC', 'AT_PREV_SC', 'HT_WL_RATIO', 'AT_WL_RATIO', 'HT_AVG_SC', 'AT_AVG_SC', 'HT_SC', 'AT_SC'] # add any new inputs and adjust outputs as neccessary (possibly errors + ELO?)
    fixtures = pd.DataFrame(columns = variables) 
    for season, group in season_groups:

        runs_for_dict = {ID : 0 for ID in unique_team_ids}
        runs_against_dict = {ID : 0 for ID in unique_team_ids}
        hits_dict = {ID : 0 for ID in unique_team_ids}
        games_played_dict = {ID : 0 for ID in unique_team_ids}
        prev_score_dict = {ID : 0 for ID in unique_team_ids}
        wins_dict = {ID : 0 for ID in unique_team_ids}
        losses_dict = {ID : 0 for ID in unique_team_ids}

        for index, row in group.iterrows():

            fixture_details = {}

            home_id = row['idHomeTeam']
            away_id = row['idAwayTeam']

            if home_id == 142040 or away_id == 142040:
                continue

            home_ABRV = get_ABRV_from_id(home_id)
            away_ABRV = get_ABRV_from_id(away_id)
            date = row['dateEvent']

            # Add all relevant information to dataframe for specific fixture
            fixture_details['HT_AT_DATE'] = home_ABRV + '_' + away_ABRV + '_' + date
            fixture_details['DATE'] = date

            fixture_details['HT'] = get_teamcode_from_id(home_id)
            fixture_details['AT'] = get_teamcode_from_id(away_id)

            fixture_details['HT_RD'] = runs_for_dict[home_id] - runs_against_dict[home_id]
            fixture_details['AT_RD'] = runs_for_dict[away_id] - runs_against_dict[away_id]

            fixture_row_elo = elo_db[(elo_db['date'] == date) & (elo_db['team1'] == home_ABRV)]
            if not len(fixture_row_elo):
                fixture_row_elo = elo_db[(elo_db['date'] == date) & (elo_db['team2'] == away_ABRV)]
                if not len(fixture_row_elo):
                    continue

            fixture_details['HT_ELO'] = fixture_row_elo['elo1_pre'].values[0]
            fixture_details['AT_ELO'] = fixture_row_elo['elo2_pre'].values[0]

            if games_played_dict[home_id] == 0:
                fixture_details['HT_HPG'] = 0
            else:
                fixture_details['HT_HPG'] = hits_dict[home_id] / games_played_dict[home_id]
                
            if games_played_dict[away_id] == 0:
                fixture_details['AT_HPG'] = 0
            else:
                fixture_details['AT_HPG'] = hits_dict[away_id] / games_played_dict[away_id]

            fixture_details['HT_PREV_SC'] = prev_score_dict[home_id]
            fixture_details['AT_PREV_SC'] = prev_score_dict[away_id]

            if losses_dict[home_id] == 0:
                fixture_details['HT_WL_RATIO'] = wins_dict[home_id] / 1
            else:
                fixture_details['HT_WL_RATIO'] = wins_dict[home_id] / losses_dict[home_id]

            if losses_dict[away_id] == 0:
                fixture_details['AT_WL_RATIO'] = wins_dict[away_id] / 1
            else:
                fixture_details['AT_WL_RATIO'] = wins_dict[away_id] / losses_dict[away_id]

            if games_played_dict[home_id] == 0:
                fixture_details['HT_AVG_SC'] = 4.34
            else:
                fixture_details['HT_AVG_SC'] = runs_for_dict[home_id] / games_played_dict[home_id]

            if games_played_dict[away_id] == 0:
                fixture_details['AT_AVG_SC'] = 4.34
            else:
                fixture_details['AT_AVG_SC'] = runs_for_dict[away_id] / games_played_dict[away_id]

            home_total_runs = row['intHomeScore']
            away_total_runs = row['intAwayScore']

            fixture_details['HT_SC'] = home_total_runs
            fixture_details['AT_SC'] = away_total_runs

            # Append stats to dictionaries so summarised statistics can be gathered
            
            runs_for_dict[home_id] += row['intHomeScore']
            runs_for_dict[away_id] += row['intAwayScore']

            runs_against_dict[home_id] += row['intAwayScore']
            runs_against_dict[away_id] += row['intHomeScore']

            fixture_result = row['strResult']
            if 'Team Totals' in fixture_result:
                temp = fixture_result.split('Team Totals')
                home_results_table = temp[1].split('\r\n\r')[0]
                away_results_table = temp[2]
                home_table_split = home_results_table.split(' ')
                home_table_split = list(filter(None, home_table_split))
                away_table_split = away_results_table.split(' ')
                away_table_split = list(filter(None, away_table_split))
                home_hits = int(home_table_split[2])
                away_hits = int(away_table_split[2])
            else:
                if 'Hits' not in fixture_result:
                    continue
                
                team_results = fixture_result.split('Hits:') # - Errors:
                home_hits_str = team_results[1].split('- Errors:')[0]
                if any(char.isdigit() for char in home_hits_str):
                    home_hits = int(home_hits_str)
                else:
                    continue
                away_hits_str = team_results[2].split('- Errors:')[0]
                if any(char.isdigit() for char in away_hits_str):
                    away_hits = int(away_hits_str)
                else:
                    continue
            
            hits_dict[home_id] += home_hits
            hits_dict[away_id] += away_hits

            games_played_dict[home_id] += 1
            games_played_dict[away_id] += 1

            prev_score_dict[home_id] = home_total_runs
            prev_score_dict[away_id] = away_total_runs

            if home_total_runs > away_total_runs:
                wins_dict[home_id] += 1
                losses_dict[away_id] += 1
            else:
                wins_dict[away_id] += 1
                losses_dict[home_id] += 1
        
            fixtures.loc[row_index] = fixture_details
            print('Fixture Added: ', fixture_details['HT_AT_DATE'])
            row_index += 1

        print('Group Complete ', season)

    file_path = 'data_and_models/mlb_model_ready_data_comb.csv'       
    fixtures.to_csv(file_path, index=False)

def get_ABRV_from_id(id:float):

    team_ABRV = {
    135278 : 'SD', 
    135269 : 'CHC',
    135272 :'LAD', 
    135264 : 'TEX',
    135271 : 'COL',
    135259 : 'MIN',
    135280 : 'STL',
    135265 : 'TOR',
    135275 : 'NYM',
    135279 : 'SF',
    135257 : 'KC',
    135261 : 'OAK',
    135253 : 'CWS',
    135258 : 'LAA',
    135267 : 'ARI',
    135251 : 'BAL',
    135254 : 'CLE',
    135255 : 'DET',
    135263 : 'TB', 
    135281 : 'WSH',
    135260 : 'NYY',
    135276 : 'PHI',
    135274 : 'MIL',
    135262 : 'SEA',
    135277 : 'PIT',
    135252 : 'BOS',
    135273 : 'MIA',
    135256 : 'HOU',  
    135270 : 'CIN',
    135268 : 'ATL' 
    }

    return team_ABRV[id]

def get_id_from_ABRV(ABRV:str):

    team_id = {
    'SD' : 135278, 
    'CHC' : 135269,
    'LAD' : 135272, 
    'TEX' : 135264,
    'COL' : 135271,
    'MIN' : 135259,
    'STL' : 135280,
    'TOR' : 135265,
    'NYM' : 135275,
    'SF' : 135279,
    'KC' : 135257,
    'OAK' : 135261,
    'CWS' : 135253,
    'LAA' : 135258,
    'ARI' : 135267,
    'BAL' : 135251,
    'CLE' : 135254,
    'DET' : 135255,
    'TB' : 135263, 
    'WSH' : 135281,
    'NYY' : 135260,
    'PHI' : 135276,
    'MIL' : 135274,
    'SEA' : 135262,
    'PIT' : 135277,
    'BOS' : 135252,
    'MIA' : 135273,
    'HOU' : 135256,  
    'CIN' : 135270,
    'ATL' : 135268 
    }

    return team_id[ABRV]

def get_teamcode_from_id(id:float):

    teamcode = {
    135260 : 30,
    135272 : 29,
    135280 : 28,
    135279 : 27,
    135268 : 26,
    135269 : 25,
    135261 : 24,
    135252 : 23,
    135255 : 22,  
    135270 : 21,
    135277 : 20,
    135276 : 19,
    135251 : 18,
    135259 : 17,
    135253 : 16,
    135254 : 15,
    135275 : 14,
    135256 : 13,
    135257 : 12,
    135264 : 11,
    135278 : 10,
    135265 : 9,
    135263 : 8,
    135273 : 7,
    135271 : 6,
    135258 : 5,
    135267 : 4,
    135281 : 3,
    135274 : 2,
    135262 : 1,
    }

    return teamcode[id]

def name_extraction():

    initial_db = pd.read_csv('mlb_fixture_details.csv')

    unique_team_ids = initial_db['idHomeTeam'].unique()
    
    filtered_db = initial_db[initial_db['idHomeTeam'].isin(unique_team_ids)]
    filtered_db = filtered_db.drop_duplicates(subset='idHomeTeam')
    result = filtered_db[['strEventAlternate', 'idHomeTeam']]
    result['strEventAlternate'] = result['strEventAlternate'].apply(lambda x: x.split('@')[1] if '@' in x else x)
    print(result)


def outliers():   
    
    file_path = 'data_and_models/mlb_model_ready_data.csv'
    data = pd.read_csv(file_path)

    ht_sc_95 = np.percentile(data['HT_SC'], 95)
    at_sc_95 = np.percentile(data['AT_SC'], 95)

    print('HT 95% quartile = ', ht_sc_95)
    print('AT 95% quartile = ', at_sc_95)

    # Plot histograms
    plt.figure(figsize=(12, 5))

    # Histogram for HT_SC
    plt.subplot(1, 2, 1)
    plt.hist(data['HT_SC'], bins=20, color='blue', alpha=0.7)
    plt.title('Histogram of HT_SC')
    plt.xlabel('HT_SC')
    plt.ylabel('Frequency')

    # Histogram for AT_SC
    plt.subplot(1, 2, 2)
    plt.hist(data['AT_SC'], bins=20, color='green', alpha=0.7)
    plt.title('Histogram of AT_SC')
    plt.xlabel('AT_SC')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

combined_db_creation()