import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class AnomalyDetector:
    def __init__(self, method='isolation_forest'):
        self.method = method
        self.scaler = StandardScaler()

    def preprocess_data(self, data):
        """Preprocess data for anomaly detection."""
        data['returns'] = data['Close'].pct_change()
        data.dropna(inplace=True)
        features = data[['returns']]
        features_scaled = self.scaler.fit_transform(features)
        return features_scaled

    def isolation_forest(self, data):
        """Isolation Forest for anomaly detection."""
        model = IsolationForest(contamination=0.05)
        model.fit(data)
        anomalies = model.predict(data)
        return anomalies

    def one_class_svm(self, data):
        """One-Class SVM for anomaly detection."""
        model = OneClassSVM(nu=0.05)
        model.fit(data)
        anomalies = model.predict(data)
        return anomalies

    def autoencoder(self, data):
        """Autoencoder for anomaly detection."""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(data.shape[1],)),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(data.shape[1], activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        model.fit(data, data, epochs=20, batch_size=32, verbose=0)
        reconstructions = model.predict(data)
        mse = np.mean(np.power(data - reconstructions, 2), axis=1)
        anomalies = np.where(mse > np.percentile(mse, 95), -1, 1)  # Top 5% as anomalies
        return anomalies

    def random_forest(self, data, labels):
        """Random Forest for supervised anomaly detection."""
        model = RandomForestClassifier(n_estimators=100)
        model.fit(data, labels)
        anomalies = model.predict(data)
        return anomalies

    def xgboost(self, data, labels):
        """XGBoost for supervised anomaly detection."""
        model = XGBClassifier(n_estimators=100)
        model.fit(data, labels)
        anomalies = model.predict(data)
        return anomalies

    def detect_anomalies(self, data, method=None, labels=None):
        """Detect anomalies using the specified method."""
        method = method or self.method
        data_scaled = self.preprocess_data(data)

        if method == 'isolation_forest':
            anomalies = self.isolation_forest(data_scaled)
        elif method == 'one_class_svm':
            anomalies = self.one_class_svm(data_scaled)
        elif method == 'autoencoder':
            anomalies = self.autoencoder(data_scaled)
        elif method == 'random_forest' and labels is not None:
            anomalies = self.random_forest(data_scaled, labels)
        elif method == 'xgboost' and labels is not None:
            anomalies = self.xgboost(data_scaled, labels)
        else:
            raise ValueError("Invalid method or missing labels for supervised methods.")

        data['anomaly'] = np.where(anomalies == -1, 1, 0)  # 1 = anomaly, 0 = normal
        return data