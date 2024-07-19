# mlbOracle

Base database files: 
mlb_elo.csv
mlb_fixture_details.csv

These combined and manipulated using 'database_creator/combined_db_creation' to make:
mlb_model_ready_data_comb.csv
which is used to build the model. 

The new predictions are then powered by the inputs retrieved from:
current_team_data.xlsx

This file is built from two data sources in particular: 
https://www.mlb.com/stats/team ( make sure stats are taken from regular season table )
+
https://neilpaine.substack.com/p/2024-mlb-elo-ratings-and-playoff?open=false#%C2%A7mlb-elo-ratings-and-win-projections ( just for ELO rating )

The current team database will need updating regularly to allow accurate predictions. Automation also viable.

* BONUS *
A classic Baseball database containing years of statistics, could be a useful extention... 
https://www.retrosheet.org/ 