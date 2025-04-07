import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="NHL Game Predictor", layout="wide")

# 🔐 Simple Login Gate
from hashlib import sha256

def check_login():
    st.sidebar.title("🔐 Login")
    user_input = st.sidebar.text_input("Username")
    pass_input = st.sidebar.text_input("Password", type="password")

    if user_input == "nhl2025" and pass_input == "Bengals1628!":
        return True
    else:
        st.sidebar.warning("Enter your credentials to access the app.")
        return False

if not check_login():
    st.stop()

TEAM_IDS = {
    'Anaheim Ducks': 24, 'Arizona Coyotes': 53, 'Boston Bruins': 6, 'Buffalo Sabres': 7,
    'Calgary Flames': 20, 'Carolina Hurricanes': 12, 'Chicago Blackhawks': 16,
    'Colorado Avalanche': 21, 'Columbus Blue Jackets': 29, 'Dallas Stars': 25,
    'Detroit Red Wings': 17, 'Edmonton Oilers': 22, 'Florida Panthers': 13,
    'Los Angeles Kings': 26, 'Minnesota Wild': 30, 'Montreal Canadiens': 8,
    'Nashville Predators': 18, 'New Jersey Devils': 1, 'New York Islanders': 2,
    'New York Rangers': 3, 'Ottawa Senators': 9, 'Philadelphia Flyers': 4,
    'Pittsburgh Penguins': 5, 'San Jose Sharks': 28, 'Seattle Kraken': 55,
    'St. Louis Blues': 19, 'Tampa Bay Lightning': 14, 'Toronto Maple Leafs': 10,
    'Vancouver Canucks': 23, 'Vegas Golden Knights': 54, 'Washington Capitals': 15,
    'Winnipeg Jets': 52
}

GOALIE_STATS = {
    'Winnipeg Jets': {'sv%': 0.924, 'gaa': 2.04, 'so': 7},
    'Tampa Bay Lightning': {'sv%': 0.922, 'gaa': 2.16, 'so': 6},
    'Dallas Stars': {'sv%': 0.912, 'gaa': 2.45, 'so': 2},
    'Washington Capitals': {'sv%': 0.910, 'gaa': 2.49, 'so': 2},
    'Florida Panthers': {'sv%': 0.906, 'gaa': 2.44, 'so': 5},
    'Minnesota Wild': {'sv%': 0.915, 'gaa': 2.55, 'so': 5},
    'Vegas Golden Knights': {'sv%': 0.907, 'gaa': 2.5, 'so': 4},
    'Montreal Canadiens': {'sv%': 0.900, 'gaa': 2.85, 'so': 4},
    'Los Angeles Kings': {'sv%': 0.922, 'gaa': 2.03, 'so': 5},
    'San Jose Sharks': {'sv%': 0.914, 'gaa': 2.49, 'so': 4},
    'New York Islanders': {'sv%': 0.905, 'gaa': 2.76, 'so': 3},
    'St. Louis Blues': {'sv%': 0.902, 'gaa': 2.69, 'so': 3},
    'Seattle Kraken': {'sv%': 0.910, 'gaa': 2.63, 'so': 2},
    'New Jersey Devils': {'sv%': 0.903, 'gaa': 2.43, 'so': 4},
    'Carolina Hurricanes': {'sv%': 0.899, 'gaa': 2.57, 'so': 2},
    'Calgary Flames': {'sv%': 0.910, 'gaa': 2.63, 'so': 3},
    'Toronto Maple Leafs': {'sv%': 0.906, 'gaa': 2.78, 'so': 1},
    'New York Rangers': {'sv%': 0.904, 'gaa': 2.86, 'so': 5},
    'Columbus Blue Jackets': {'sv%': 0.890, 'gaa': 3.24, 'so': 1},
    'Utah Hockey Club': {'sv%': 0.906, 'gaa': 2.53, 'so': 1},
    'Vancouver Canucks': {'sv%': 0.899, 'gaa': 2.66, 'so': 4},
    'Edmonton Oilers': {'sv%': 0.894, 'gaa': 2.91, 'so': 2},
    'Buffalo Sabres': {'sv%': 0.885, 'gaa': 3.23, 'so': 2},
    'Ottawa Senators': {'sv%': 0.909, 'gaa': 2.75, 'so': 3},
    'Anaheim Ducks': {'sv%': 0.903, 'gaa': 3.07, 'so': 1},
    'Boston Bruins': {'sv%': 0.895, 'gaa': 3.08, 'so': 4},
    'Philadelphia Flyers': {'sv%': 0.881, 'gaa': 3.15, 'so': 2},
    'Nashville Predators': {'sv%': 0.895, 'gaa': 2.99, 'so': 4},
    'Detroit Red Wings': {'sv%': 0.896, 'gaa': 2.84, 'so': 1},
    'Pittsburgh Penguins': {'sv%': 0.888, 'gaa': 3.26, 'so': 1},
    'Chicago Blackhawks': {'sv%': 0.891, 'gaa': 3.35, 'so': 1},
    'Colorado Avalanche': {'sv%': 0.873, 'gaa': 3.69, 'so': 0}
}

@st.cache_data

def load_data():
    default_url = "https://raw.githubusercontent.com/OpenNHL/statistics/main/RegularSeason/games.csv"
    st.sidebar.markdown("### 📥 Add CSV Data URLs")
    url_input = st.sidebar.text_area("Enter one or more CSV URLs (one per line):", value=default_url, height=150)
    urls = [url.strip() for url in url_input.splitlines() if url.strip()]
    all_frames = []
    for url in urls:
        try:
            df = pd.read_csv(url)
            df = df[['date', 'homeTeam', 'awayTeam', 'homeGoals', 'awayGoals']]
            df.dropna(inplace=True)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df.dropna(subset=['date'], inplace=True)
            df['homeWin'] = (df['homeGoals'] > df['awayGoals']).astype(int)
            all_frames.append(df)
        except Exception as e:
            st.sidebar.error(f"Failed to load {url}: {e}")
    if not all_frames:
        st.error("No valid data loaded. Please check your URLs.")
        st.stop()
    return pd.concat(all_frames, ignore_index=True)

def get_team_stats(team_name):
    team_id = TEAM_IDS.get(team_name)
    if team_id is None:
        return None
    url = f"https://statsapi.web.nhl.com/api/v1/teams/{team_id}/stats"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        stats = response.json()['stats'][0]['splits'][0]['stat']
        goalie_stats = GOALIE_STATS.get(team_name, {'sv%': 0.9, 'gaa': 2.9, 'so': 1})
        return {
            'goalsPerGame': float(stats['goalsPerGame']),
            'goalsAgainstPerGame': float(stats['goalsAgainstPerGame']),
            'powerPlayPercentage': float(stats['powerPlayPercentage']),
            'penaltyKillPercentage': float(stats['penaltyKillPercentage']),
            'savePercentage': goalie_stats['sv%'],
            'goalsAgainstAverage': goalie_stats['gaa'],
            'shutouts': goalie_stats['so']
        }
    except:
        return None
    url = f"https://statsapi.web.nhl.com/api/v1/teams/{team_id}/stats"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        stats = response.json()['stats'][0]['splits'][0]['stat']
        return {
            'goalsPerGame': float(stats['goalsPerGame']),
            'goalsAgainstPerGame': float(stats['goalsAgainstPerGame']),
            'powerPlayPercentage': float(stats['powerPlayPercentage']),
            'penaltyKillPercentage': float(stats['penaltyKillPercentage']),
            'savePercentage': GOALIE_SAVE_PCT.get(team_name, 0.9)
        }
    except:
        return None
    url = f"https://statsapi.web.nhl.com/api/v1/teams/{team_id}/stats"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        stats = response.json()['stats'][0]['splits'][0]['stat']
        return {
            'goalsPerGame': float(stats['goalsPerGame']),
            'goalsAgainstPerGame': float(stats['goalsAgainstPerGame']),
            'powerPlayPercentage': float(stats['powerPlayPercentage']),
            'penaltyKillPercentage': float(stats['penaltyKillPercentage']),
            'savePercentage': GOALIE_SAVE_PCT.get(team_name, 0.9)
        }
    except:
        return None
    team_url = f"https://statsapi.web.nhl.com/api/v1/teams/{team_id}/stats"
    goalie_url = f"https://statsapi.web.nhl.com/api/v1/teams/{team_id}/roster?position=G"
    try:
        # Team stats
        response = requests.get(team_url, timeout=10)
        response.raise_for_status()
        stats = response.json()['stats'][0]['splits'][0]['stat']

        # Goalie stats (avg save % of current rostered goalies)
        goalie_response = requests.get(goalie_url, timeout=10)
        goalie_response.raise_for_status()
        goalies = goalie_response.json()['roster']
        sv_percentages = []
        for g in goalies:
            player_id = g['person']['id']
            player_stats_url = f"https://statsapi.web.nhl.com/api/v1/people/{player_id}/stats?stats=statsSingleSeason"
            stat_resp = requests.get(player_stats_url)
            if stat_resp.status_code == 200:
                data = stat_resp.json()
                try:
                    goalie_sv = float(data['stats'][0]['splits'][0]['stat']['savePercentage'])
                    sv_percentages.append(goalie_sv)
                except:
                    continue

        avg_sv_pct = round(np.mean(sv_percentages), 3) if sv_percentages else 0.9

        return {
            'goalsPerGame': float(stats['goalsPerGame']),
            'goalsAgainstPerGame': float(stats['goalsAgainstPerGame']),
            'powerPlayPercentage': float(stats['powerPlayPercentage']),
            'penaltyKillPercentage': float(stats['penaltyKillPercentage']),
            'savePercentage': avg_sv_pct
        }
    except:
        return None
    url = f"https://statsapi.web.nhl.com/api/v1/teams/{team_id}/stats"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        stats = response.json()['stats'][0]['splits'][0]['stat']
        return {
            'goalsPerGame': float(stats['goalsPerGame']),
            'goalsAgainstPerGame': float(stats['goalsAgainstPerGame']),
            'powerPlayPercentage': float(stats['powerPlayPercentage']),
            'penaltyKillPercentage': float(stats['penaltyKillPercentage'])
        }
    except:
        return None

def build_features(df):
    df_sorted = df.sort_values('date')
    team_stats = {}
    features = []
    for _, row in df_sorted.iterrows():
        home, away = row['homeTeam'], row['awayTeam']
        home_stats = team_stats.get(home, {'gf': [], 'ga': [], 'pp': [], 'pk': [], 'sv%': []})
        away_stats = team_stats.get(away, {'gf': [], 'ga': [], 'pp': [], 'pk': [], 'sv%': []})

        def weighted_avg(values):
            weights = [4 if i >= len(values) - 5 else 3 if i >= len(values) - 10 else 2 if i >= len(values) - 15 else 1 for i in range(len(values))]
            return np.average(values, weights=weights[-len(values):]) if values else 3

        features.append({
            'home_gaa': weighted_avg(home_stats.get('gaa', [])),
            'away_gaa': weighted_avg(away_stats.get('gaa', [])),
            'home_so': weighted_avg(home_stats.get('so', [])),
            'away_so': weighted_avg(away_stats.get('so', [])),
            'home_avg_gf': weighted_avg(home_stats['gf']),
            'home_avg_ga': weighted_avg(home_stats['ga']),
            'away_avg_gf': weighted_avg(away_stats['gf']),
            'away_avg_ga': weighted_avg(away_stats['ga']),
            'home_pp': weighted_avg(home_stats['pp']),
            'home_pk': weighted_avg(home_stats['pk']),
            'away_pp': weighted_avg(away_stats['pp']),
            'away_pk': weighted_avg(away_stats['pk']),
            'home_sv%': weighted_avg(home_stats['sv%']),
            'away_sv%': weighted_avg(away_stats['sv%']),
            'homeWin': row['homeWin']
        })

        # Real stats filled above — no longer using random data
        team_stats[home] = home_stats
        team_stats[away] = away_stats

    return pd.DataFrame(features)

# Load and prepare model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

model_options = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

model_choice = st.sidebar.selectbox("Select Prediction Model", list(model_options.keys()))
model = model_options[model_choice]
df = load_data()
feature_df = build_features(df)
X = feature_df.drop('homeWin', axis=1)
y = feature_df['homeWin']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
model_accuracy = accuracy_score(y_test, model.predict(X_test))
st.sidebar.metric(label="Model Accuracy", value=f"{model_accuracy:.2%}")

# Dashboard UI
st.title("🏒 NHL Game Predictor Dashboard")
st.markdown("Enter two teams to predict the outcome, or view today's full game predictions.")

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Select Home Team", list(TEAM_IDS.keys()))
with col2:
    away_team = st.selectbox("Select Away Team", list(TEAM_IDS.keys()))

if st.button("Predict Matchup"):
    home_stats = get_team_stats(home_team)
    away_stats = get_team_stats(away_team)
    if home_stats and away_stats:
        sample = pd.DataFrame([{
            'home_avg_gf': home_stats['goalsPerGame'],
            'home_avg_ga': home_stats['goalsAgainstPerGame'],
            'away_avg_gf': away_stats['goalsPerGame'],
            'away_avg_ga': away_stats['goalsAgainstPerGame'],
            'home_pp': home_stats['powerPlayPercentage'],
            'home_pk': home_stats['penaltyKillPercentage'],
            'away_pp': away_stats['powerPlayPercentage'],
            'away_pk': away_stats['penaltyKillPercentage'],
            'home_sv%': home_stats['savePercentage'],
            'away_sv%': away_stats['savePercentage'],
            'home_gaa': home_stats['goalsAgainstAverage'],
            'away_gaa': away_stats['goalsAgainstAverage'],
            'home_so': home_stats['shutouts'],
            'away_so': away_stats['shutouts']
        }])
        pred = model.predict(sample)[0]
        prob = model.predict_proba(sample)[0][1]
        result = f"🏠 {home_team} Win" if pred == 1 else f"✈️ {away_team} Win"
        st.success(f"**Prediction:** {result}")
        st.info(f"Confidence: {prob * 100:.2f}%")
        st.write(f"{home_team} - GF/GP: {home_stats['goalsPerGame']}, GA/GP: {home_stats['goalsAgainstPerGame']}, PP%: {home_stats['powerPlayPercentage']}, PK%: {home_stats['penaltyKillPercentage']}, SV%: {home_stats['savePercentage']}")
        st.write(f"{away_team} - GF/GP: {away_stats['goalsPerGame']}, GA/GP: {away_stats['goalsAgainstPerGame']}, PP%: {away_stats['powerPlayPercentage']}, PK%: {away_stats['penaltyKillPercentage']}, SV%: {away_stats['savePercentage']}")
    else:
        st.error("Failed to fetch stats for one or both teams.")

# Today's Game Predictions
st.markdown("---")
st.subheader("📅 Today's Matchups and Predictions")

today = datetime.date.today().isoformat()
schedule_url = f"https://statsapi.web.nhl.com/api/v1/schedule?date={today}"
schedule_response = requests.get(schedule_url).json()

results = []
for game in schedule_response.get("dates", [])[0].get("games", []):
    home_team = game['teams']['home']['team']['name']
    away_team = game['teams']['away']['team']['name']
    home_stats = get_team_stats(home_team)
    away_stats = get_team_stats(away_team)
    if home_stats and away_stats:
        sample = pd.DataFrame([{
            'home_avg_gf': home_stats['goalsPerGame'],
            'home_avg_ga': home_stats['goalsAgainstPerGame'],
            'away_avg_gf': away_stats['goalsPerGame'],
            'away_avg_ga': away_stats['goalsAgainstPerGame']
        }])
        pred = model.predict(sample)[0]
        prob = model.predict_proba(sample)[0][1]
        predicted = home_team if pred == 1 else away_team
        results.append({
            'Game': f"{away_team} @ {home_team}",
            'Home SV%': home_stats['savePercentage'],
            'Away SV%': away_stats['savePercentage'],
            'Home PP%': home_stats['powerPlayPercentage'],
            'Away PP%': away_stats['powerPlayPercentage'],
            'Home PK%': home_stats['penaltyKillPercentage'],
            'Away PK%': away_stats['penaltyKillPercentage'],
            'Home SV%': home_stats['savePercentage'],
            'Away SV%': away_stats['savePercentage'],
            'Predicted Winner': predicted,
            'Confidence (%)': round(prob * 100, 2)
        })

if results:
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='Confidence (%)', ascending=False)
    st.dataframe(df_results)
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=df_results.to_csv(index=False).encode('utf-8'),
        file_name=f'nhl_predictions_{today}.csv',
        mime='text/csv'
    )

# 🔍 Feature Importance
st.markdown("---")
st.subheader("🔍 Feature Importance")

if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    features = X.columns
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=importances, y=features, ax=ax)
    ax.set_title("Which Stats Influence Predictions Most")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")
    st.pyplot(fig)
elif hasattr(model, 'coef_'):
    importances = model.coef_[0]
    features = X.columns
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=np.abs(importances), y=features, ax=ax)
    ax.set_title("Logistic Regression Coefficients (Absolute Values)")
    ax.set_xlabel("Coefficient Magnitude")
    ax.set_ylabel("Feature")
    st.pyplot(fig)
else:
    st.info("Selected model does not support feature importance visualization.")

# Accuracy over time
if os.path.exists("predictions_log.csv"):
    st.markdown("---")
    st.subheader("📊 Accuracy Over Time")
    log_df = pd.read_csv("predictions_log.csv")
    tracked = log_df.dropna(subset=['correct'])
    if not tracked.empty:
        tracked['cumulative_accuracy'] = tracked['correct'].expanding().mean()
        fig, ax = plt.subplots()
        sns.lineplot(data=tracked, x=tracked.index, y='cumulative_accuracy', ax=ax)
        ax.set_title("Cumulative Prediction Accuracy")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.grid(True)
        st.pyplot(fig)
