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

st.sidebar.markdown("### 📥 Add CSV Data URLs")
default_url = "https://raw.githubusercontent.com/Aburrows28/Cinco/main/example_games.csv"
