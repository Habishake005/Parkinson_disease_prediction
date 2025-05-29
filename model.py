import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
import xgboost as xgb

# Load dataset
df = pd.read_csv('parkinsons_updrs.data')
voice_features = df.columns[6:-2]
X = df[voice_features].values
y = df['motor_UPDRS'].values

# Feature selection
rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)
selected_columns = df[voice_features].columns[rfe.get_support()]

# Create sequences
T = 20
X_seq, y_seq = [], []
for i in range(len(X_selected) - T):
    X_seq.append(X_selected[i:i+T])
    y_seq.append(y[i+T])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Custom Attention Layer
class Attention(layers.Layer):
    def call(self, inputs):
        score = tf.nn.tanh(inputs)
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context

# LSTM+Attention model
def build_lstm_attention_model(input_shape):
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        Attention(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss=tf.keras.losses.Huber())
    return model

# TCN block
class TCNBlock(layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = layers.Conv1D(filters, kernel_size, padding='causal',
                                   dilation_rate=dilation_rate, activation='relu')
        self.dropout1 = layers.Dropout(dropout)
        self.conv2 = layers.Conv1D(filters, kernel_size, padding='causal',
                                   dilation_rate=dilation_rate, activation='relu')
        self.dropout2 = layers.Dropout(dropout)
        self.downsample = layers.Conv1D(filters, 1)
        self.activation = layers.Activation('relu')

    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.dropout2(x, training=training)
        res = self.downsample(inputs)
        return self.activation(x + res)

# TCN model
def build_tcn_model(input_shape, num_filters=64, kernel_size=3, num_blocks=3, dropout=0.2):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for i in range(num_blocks):
        x = TCNBlock(num_filters, kernel_size, dilation_rate=2 ** i, dropout=dropout)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss=tf.keras.losses.Huber())
    return model

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
T, features = X_seq.shape[1], X_seq.shape[2]
lstm_preds_all, tcn_preds_all, y_true_all = [], [], []
results = {'lstm_attention': {'mse': [], 'r2': []}, 'tcn': {'mse': [], 'r2': []}}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_seq), 1):
    print(f"\n🔁 Fold {fold}/5")
    X_train, X_val = X_seq[train_idx], X_seq[val_idx]
    y_train, y_val = y_seq[train_idx], y_seq[val_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, features)).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, features)).reshape(X_val.shape)

    # LSTM+Attention
    lstm_model = build_lstm_attention_model((T, features))
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                   epochs=100, batch_size=64, callbacks=[early_stop], verbose=0)
    y_pred_lstm = lstm_model.predict(X_val).flatten()
    results['lstm_attention']['mse'].append(mean_squared_error(y_val, y_pred_lstm))
    results['lstm_attention']['r2'].append(r2_score(y_val, y_pred_lstm))
    lstm_preds_all.extend(y_pred_lstm)

    # TCN
    tcn_model = build_tcn_model((T, features))
    tcn_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=100, batch_size=64, callbacks=[early_stop], verbose=0)
    y_pred_tcn = tcn_model.predict(X_val).flatten()
    results['tcn']['mse'].append(mean_squared_error(y_val, y_pred_lstm))
    results['tcn']['r2'].append(r2_score(y_val, y_pred_lstm))
    tcn_preds_all.extend(y_pred_tcn)

    y_true_all.extend(y_val)

print("\n===== Final Cross-Validation Results =====")
print(f"LSTM+Attention Mean MSE: {np.mean(results['lstm_attention']['mse']):.4f}")
print(f"LSTM+Attention Mean R²: {np.mean(results['lstm_attention']['r2']):.4f}")
print(f"TCN Mean MSE: {np.mean(results['tcn']['mse']):.4f}")
print(f"TCN Mean R²: {np.mean(results['tcn']['r2']):.4f}")
# Meta model (XGBoost)
X_meta = np.column_stack([lstm_preds_all, tcn_preds_all])
y_meta = np.array(y_true_all)
meta_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
meta_model.fit(X_meta, y_meta)
meta_preds = meta_model.predict(X_meta)

# Final metrics
mse = mean_squared_error(y_meta, meta_preds)
r2 = r2_score(y_meta, meta_preds)
print("\n===== Meta-Model Performance =====")
print(f"Meta Model MSE: {mse:.4f}")
print(f"Meta Model R²: {r2:.4f}")


