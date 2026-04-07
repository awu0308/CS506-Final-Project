"""
NBA Career Longevity Predictor - Data Collection 
"""

import pandas as pd
import time
from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.stats.static import players


MIN_MINUTES_PER_GAME = 15.0
MIN_GAMES = 20
MIN_PIE = 0.07 
START_SEASON_YEAR = 2000
END_SEASON_YEAR = 2024

# Pull season-by-season league stats in bulk
print("Pulling league-wide stats season by season")

all_seasons = []

for year in range(START_SEASON_YEAR, END_SEASON_YEAR + 1):
    season_str = f"{year}-{str(year + 1)[-2:]}"  # e.g., "2000-01"
    print(f"  Fetching {season_str}...")

    try:
        stats_base = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season_str,
            per_mode_detailed='PerGame',
            measure_type_detailed_defense='Base'
        )
        df_base = stats_base.get_data_frames()[0]
        time.sleep(1)

        stats_adv = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season_str,
            per_mode_detailed='PerGame',
            measure_type_detailed_defense='Advanced'
        )
        df_adv = stats_adv.get_data_frames()[0][['PLAYER_ID', 'PIE']]
        time.sleep(1)

        df = df_base.merge(df_adv, on='PLAYER_ID', how='left')
        df['SEASON'] = season_str
        df['SEASON_START_YEAR'] = year
        all_seasons.append(df)
    except Exception as e:
        print(f"    Error on {season_str}: {e}")
        time.sleep(3)
        continue

print(f"\n  Fetched {len(all_seasons)} seasons")

# Combine and clean
print("Combining and cleaning")

df_all = pd.concat(all_seasons, ignore_index=True)

df_all = df_all.rename(columns={
    'PLAYER_ID': 'PLAYER_ID',
    'PLAYER_NAME': 'PLAYER_NAME',
    'GP': 'GP',
    'MIN': 'MPG',    
    'PTS': 'PPG',
    'REB': 'RPG',
    'AST': 'APG',
})

print(f"  Total player-season records: {len(df_all)}")
print(f"  Unique players: {df_all['PLAYER_ID'].nunique()}")

# STEP 3: Calculate total minutes 
# MPG * GP gives approximate total minutes
df_all['TOTAL_MIN'] = df_all['MPG'] * df_all['GP']

# Sort by player and season
df_all = df_all.sort_values(['PLAYER_ID', 'SEASON_START_YEAR'])

# Years in league
first_seasons = df_all.groupby('PLAYER_ID')['SEASON_START_YEAR'].min().reset_index()
first_seasons.columns = ['PLAYER_ID', 'FIRST_SEASON_YEAR']
df_all = df_all.merge(first_seasons, on='PLAYER_ID')
df_all['YEARS_IN_LEAGUE'] = df_all['SEASON_START_YEAR'] - df_all['FIRST_SEASON_YEAR']

# Label competitive seasons
print("\nLabeling competitive seasons")

df_all['IS_COMPETITIVE'] = (
    (df_all['MPG'] >= MIN_MINUTES_PER_GAME) &
    (df_all['GP'] >= MIN_GAMES) &
    (df_all['PIE'] >= MIN_PIE)
)

print(f"  Competitive seasons: {df_all['IS_COMPETITIVE'].sum()}")
print(f"  Non-competitive seasons: {(~df_all['IS_COMPETITIVE']).sum()}")

# Calculate remaining competitive seasons
print("Calculating remaining competitive seasons")

df_all = df_all.sort_values(['PLAYER_ID', 'SEASON_START_YEAR']).reset_index(drop=True)

total_comp = df_all.groupby('PLAYER_ID')['IS_COMPETITIVE'].transform('sum')
cum_comp = df_all.groupby('PLAYER_ID')['IS_COMPETITIVE'].cumsum()
df_all['REMAINING_COMPETITIVE_SEASONS'] = (total_comp - cum_comp).astype(int)

# Feature engineering
print("Engineering features")

df_all = df_all.sort_values(['PLAYER_ID', 'SEASON_START_YEAR'])

# Cumulative stats
df_all['CUMULATIVE_MINUTES'] = df_all.groupby('PLAYER_ID')['TOTAL_MIN'].cumsum()
df_all['CUMULATIVE_GAMES'] = df_all.groupby('PLAYER_ID')['GP'].cumsum()

# Peak tracking
df_all['CAREER_PEAK_PPG'] = df_all.groupby('PLAYER_ID')['PPG'].cummax()
df_all['DISTANCE_FROM_PEAK'] = df_all['CAREER_PEAK_PPG'] - df_all['PPG']

# Season-over-season changes
df_all['PPG_CHANGE'] = df_all.groupby('PLAYER_ID')['PPG'].diff()
df_all['MPG_CHANGE'] = df_all.groupby('PLAYER_ID')['MPG'].diff()

print(f"  Final dataset shape: {df_all.shape}")

# Output
output_path = "nba_career_longevity_data.csv"
df_all.to_csv(output_path, index=False)
print(f"\nDone! Saved to {output_path}")

print("\n--- Summary ---")
print(f"Players: {df_all['PLAYER_ID'].nunique()}")
print(f"Seasons covered: {df_all['SEASON_START_YEAR'].min()} to {df_all['SEASON_START_YEAR'].max()}")
print(f"Total records: {len(df_all)}")
print(f"\nRemaining competitive seasons distribution:")
print(df_all['REMAINING_COMPETITIVE_SEASONS'].describe())
