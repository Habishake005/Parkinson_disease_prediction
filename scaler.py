# -*- coding: utf-8 -*-
"""
Created on Thu May 22 08:51:53 2025

@author: Asus
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
import pickle
import gzip

# Load your dataset
df = pd.read_csv(r'C:\Users\Asus\Downloads\parkinsons_updrs.data')

# Extract voice features and target
voice_features = df.columns[6:-2]
X = df[voice_features].values
y = df['motor_UPDRS'].values

# Feature selection using RFE (same as before)
rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)

# Fit the scaler
scaler = StandardScaler()
scaler.fit(X_selected)

# Save the scaler to a compressed file
with gzip.open('scaler.gz', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Scaler has been refitted and saved as 'scaler.gz'")
