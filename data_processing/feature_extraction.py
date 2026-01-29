import pandas as pd
from tqdm import tqdm

# check if match should be processed
def is_match_eligible(match, existing_match_ids, start_season, start_matchweek):
    if match['match_id'] in existing_match_ids:
        return False
        
    if match['season'] < start_season:
        return False
    
    # to start from the 6th GW ONLY in 2023 (start of available data)
    if match['season'] == start_season and match['matchweek'] < start_matchweek:
        return False
    
    return True

def compute_team_rolling_features(historical_data, team, window):
    team_matches = historical_data[
        (historical_data['home_team'] == team) |
        (historical_data['away_team'] == team)
    ].tail(window)

    points = 0
    goals_for = 0
    goals_against = 0

    for _, game in team_matches.iterrows():
        if game['home_team'] == team:
            goals_for += game['home_score']
            goals_against += game['away_score']

            if game['winner'] == 'HOME_TEAM':
                points += 3
            elif game['winner'] == 'DRAW':
                points += 1
        else:
            goals_for += game['away_score']
            goals_against += game['home_score']

            if game['winner'] == 'AWAY_TEAM':
                points += 3
            elif game['winner'] == 'DRAW':
                points += 1

    return points, goals_for, goals_against

def compute_venue_averages(historical_data, team, venue):
    if venue not in {"home", "away"}:
        raise ValueError("venue must be 'home' or 'away'")

    if venue == "home":
        team_col = "home_team"
        goals_for_col = "home_score"
        goals_against_col = "away_score"
    else:
        team_col = "away_team"
        goals_for_col = "away_score"
        goals_against_col = "home_score"

    venue_matches = historical_data[historical_data[team_col] == team]

    if len(venue_matches) == 0:
        return 0.0, 0.0

    goals_for_avg = venue_matches[goals_for_col].mean()
    goals_against_avg = venue_matches[goals_against_col].mean()

    return goals_for_avg, goals_against_avg

def compute_h2h_features(historical_data, home_team, away_team, window):
    h2h_matches = historical_data[
            ((historical_data['home_team'] == home_team) & (historical_data['away_team'] == away_team)) |
            ((historical_data['home_team'] == away_team) & (historical_data['away_team'] == home_team))
        ].tail(window)
    
    h2h_home_points_last5 = 0
    h2h_away_points_last5 = 0
    
    for _, game in h2h_matches.iterrows():
            if game['home_team'] == home_team:
                if game['winner'] == 'HOME_TEAM':
                    h2h_home_points_last5 += 3
                elif game['winner'] == 'DRAW':
                    h2h_home_points_last5 += 1
                    h2h_away_points_last5 += 1
                else:
                    h2h_away_points_last5 += 3
            else:
                if game['winner'] == 'AWAY_TEAM':
                    h2h_home_points_last5 += 3
                elif game['winner'] == 'DRAW':
                    h2h_home_points_last5 += 1
                    h2h_away_points_last5 += 1
                else:
                    h2h_away_points_last5 += 3
    
    return h2h_home_points_last5,h2h_away_points_last5

def compute_is_big_game(home_team, away_team, derby_groups, historic_top6):
    is_derby = 0
    for derby_group in derby_groups:
        if home_team in derby_group['teams'] and away_team in derby_group['teams']:
            is_derby = derby_group['weight']
            break
    
    num_top6 = (1 if home_team in historic_top6 else 0) + (1 if away_team in historic_top6 else 0)
    return 2*is_derby + num_top6

def build_feature_row(match, computed_features):
    feature_row = {
        "match_id": match["match_id"],
        "date": match["date"],
        "matchweek": match["matchweek"],
        "home_team": match["home_team"],
        "away_team": match["away_team"],
        **computed_features,
        "home_goals": match["home_score"],
        "away_goals": match["away_score"]
    }

    return feature_row

def compute_features_for_match(historical_data, match, rolling_window, derby_groups, historic_top6):
    home_team = match["home_team"]
    away_team = match["away_team"]

    home_points_last5, home_goals_for_last5, home_goals_against_last5 = compute_team_rolling_features(historical_data, home_team, rolling_window)
    away_points_last5, away_goals_for_last5, away_goals_against_last5 = compute_team_rolling_features(historical_data, away_team, rolling_window)

    home_home_goals_for_avg, home_home_goals_against_avg = compute_venue_averages(historical_data, home_team, "home")
    away_away_goals_for_avg, away_away_goals_against_avg = compute_venue_averages(historical_data, away_team, "away")

    h2h_home_points_last5, h2h_away_points_last5 = compute_h2h_features(historical_data, home_team, away_team, rolling_window)
    is_big_game = compute_is_big_game(home_team, away_team, derby_groups, historic_top6)

    return {
        "home_points_last5": home_points_last5,
        "away_points_last5": away_points_last5,
        "home_goals_for_last5": home_goals_for_last5,
        "away_goals_for_last5": away_goals_for_last5,
        "home_goals_against_last5": home_goals_against_last5,
        "away_goals_against_last5": away_goals_against_last5,
        "home_home_goals_for_avg": home_home_goals_for_avg,
        "home_home_goals_against_avg": home_home_goals_against_avg,
        "away_away_goals_for_avg": away_away_goals_for_avg,
        "away_away_goals_against_avg": away_away_goals_against_avg,
        "h2h_home_points_last5": h2h_home_points_last5,
        "h2h_away_points_last5": h2h_away_points_last5,
        "is_big_game": is_big_game,
    }



def extract_and_append_features(df_flat, features_path="data/features/features.parquet", start_season=2023, start_matchweek=6):
    HISTORIC_TOP6 = [
        "Arsenal FC",
        "Chelsea FC",
        "Liverpool FC",
        "Manchester City FC",
        "Manchester United FC",
        "Tottenham Hotspur FC"
    ]
    
    DERBY_GROUPS = [
        {"teams": ["Arsenal FC", "Chelsea FC", "Tottenham Hotspur FC"], "weight": 1.0},
        {"teams": ["Crystal Palace FC", "Arsenal FC", "Chelsea FC"], "weight": 0.5},
        {"teams": ["Liverpool FC", "Everton FC"], "weight": 1.0},
        {"teams": ["Manchester United FC", "Manchester City FC"], "weight": 1.0},
        {"teams": ["Newcastle United FC", "Sunderland AFC"], "weight": 0.8},
    ]
    
    ROLLING_WINDOW = 5
    
    # load existing features if they exist
    try:
        existing_features = pd.read_parquet(features_path)
        existing_match_ids = set(existing_features['match_id'].values)
        print(f"Loaded {len(existing_features)} existing features")
    except FileNotFoundError:
        existing_features = pd.DataFrame()
        existing_match_ids = set()
        print("No existing features found, creating new file")
    
    # sort by date 
    df = df_flat.sort_values('date').reset_index(drop=True)
    
    # infer season from date
    if 'season' not in df.columns:
        df['season'] = df['date'].dt.year
        # adjust for seasons that span years (e.g., Aug 2023 is 2023-24 season)
        df.loc[df['date'].dt.month >= 8, 'season'] = df['date'].dt.year
        df.loc[df['date'].dt.month < 8, 'season'] = df['date'].dt.year - 1
    
    features_list = []
    
    # process each match
    for idx, match in tqdm(df.iterrows(), total=len(df), desc="Processing matches"):
        if not is_match_eligible(match, existing_match_ids, start_season, start_matchweek):
            continue

        # skip rare edge cases
        if (pd.isna(match['home_score']) or pd.isna(match['away_score']) or
            match['home_score'] < 0 or match['away_score'] < 0):
            continue

        # get all matches before this one
        historical_data = df[df['date'] < match['date']]
        
        if len(historical_data) == 0:
            continue
        
        computed_features = compute_features_for_match(historical_data, match, ROLLING_WINDOW, DERBY_GROUPS, HISTORIC_TOP6)
        feature_row = build_feature_row(match, computed_features)
        features_list.append(feature_row)
    
    new_features_df = pd.DataFrame(features_list)
    
    print(f"Extracted {len(new_features_df)} new features")
    
    # Combine with existing features
    if len(existing_features) > 0:
        combined_features = pd.concat([existing_features, new_features_df], ignore_index=True)
    else:
        combined_features = new_features_df
    
    # Sort by date
    combined_features = combined_features.sort_values('date').reset_index(drop=True)
    
    # Save back to parquet
    combined_features.to_parquet(features_path, index=False)
    print(f"Saved {len(combined_features)} total features to {features_path}")
    
    return combined_features

if __name__ == "__main__":
    df_2023 = pd.read_parquet("data/processed/matches_flat_2023.parquet")
    df_2024 = pd.read_parquet("data/processed/matches_flat_2024.parquet")
    df_2025 = pd.read_parquet("data/processed/matches_flat_2025.parquet")
    
    # combine all data
    df_all = pd.concat([df_2023, df_2024, df_2025], ignore_index=True)
    
    features = extract_and_append_features(
        df_all, 
        features_path="data/features/features.parquet",
        start_season=2023,
        start_matchweek=6
    )
    
    print(f"\nFeature extraction complete!")
    print(f"Total features: {len(features)}")
    print(f"\nSample features:\n{features.tail()}")
