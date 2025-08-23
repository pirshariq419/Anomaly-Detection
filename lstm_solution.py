"""
Multivariate Time Series Anomaly Detection using LSTM Autoencoders (PEP8)
Author: Mohammad Shariq Pir
Date: August 23, 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import warnings

warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, RepeatVector
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Please install with: pip install tensorflow")


class LSTMAnomalyDetector:
    """
    LSTM Autoencoder-based anomaly detection for multivariate time series data.
    """

    def __init__(
        self,
        sequence_length: int = 24,
        encoding_dim: int = 64,
        lstm_units: int = 128,
        learning_rate: float = 0.001
    ):
        self.sequence_length = sequence_length
        self.encoding_dim = encoding_dim
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_names = []

    def _build_model(self, n_features: int) -> Model:
        """
        Build LSTM Autoencoder model.
        """
        inputs = Input(shape=(self.sequence_length, n_features))
        encoded = LSTM(
            self.lstm_units, activation='relu', return_sequences=True
        )(inputs)
        encoded = LSTM(
            self.encoding_dim, activation='relu', return_sequences=False
        )(encoded)
        decoded = RepeatVector(self.sequence_length)(encoded)
        decoded = LSTM(
            self.lstm_units, activation='relu', return_sequences=True
        )(decoded)
        decoded = LSTM(
            n_features, activation='linear', return_sequences=True
        )(decoded)
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        return autoencoder

    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Create sequences for LSTM input.
        """
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
        return np.array(sequences)

    def fit(
        self,
        X_train: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1
    ) -> None:
        """
        Train the LSTM Autoencoder on normal data.
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_seq = self._create_sequences(X_train_scaled)
        self.model = self._build_model(X_train.shape[1])
        self.model.fit(
            X_train_seq, X_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )

    def predict_anomaly_scores(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate anomaly scores for input data.
        """
        X_scaled = self.scaler.transform(X)
        X_seq = self._create_sequences(X_scaled)
        X_pred = self.model.predict(X_seq, verbose=0)
        reconstruction_errors = np.mean(
            np.square(X_seq - X_pred), axis=(1, 2)
        )
        feature_errors = np.mean(
            np.square(X_seq - X_pred), axis=(0, 1)
        )
        anomaly_scores = np.zeros(len(X))
        if len(reconstruction_errors) > 0:
            anomaly_scores[self.sequence_length - 1:] = reconstruction_errors
        for i in range(min(self.sequence_length - 1, len(X))):
            if i == 0:
                anomaly_scores[i] = 0.0
            else:
                partial_data = X_scaled[:i + 1]
                if len(partial_data) > 1:
                    recent_mean = np.mean(partial_data, axis=0)
                    current_deviation = np.mean(
                        np.square(X_scaled[i] - recent_mean)
                    )
                    anomaly_scores[i] = current_deviation
                else:
                    anomaly_scores[i] = 0.0
        return anomaly_scores, feature_errors

    def get_top_features(
        self, X: np.ndarray, feature_names: List[str], n_top: int = 7
    ) -> List[List[str]]:
        """
        Get top contributing features for each data point.
        """
        X_scaled = self.scaler.transform(X)
        top_features_list = []

        if len(X) >= self.sequence_length:
            X_seq = self._create_sequences(X_scaled)
            X_pred = self.model.predict(X_seq, verbose=0)
            feature_errors_per_seq = np.mean(
                np.square(X_seq - X_pred), axis=1
            )
        else:
            feature_errors_per_seq = np.array([])

        for i in range(len(X)):
            if i < self.sequence_length - 1 or len(feature_errors_per_seq) == 0:
                if i == 0:
                    feature_scores = np.ones(len(feature_names))
                else:
                    recent_data = X_scaled[:i + 1]
                    if len(recent_data) > 1:
                        feature_scores = np.var(recent_data, axis=0)
                    else:
                        feature_scores = np.abs(X_scaled[i])
            else:
                seq_idx = i - self.sequence_length + 1
                if seq_idx < len(feature_errors_per_seq):
                    feature_scores = feature_errors_per_seq[seq_idx]
                else:
                    feature_scores = np.ones(len(feature_names))

            if len(feature_scores) == len(feature_names):
                top_indices = np.argsort(feature_scores)[::-1]
                top_features = []
                for idx in top_indices:
                    if len(top_features) < n_top:
                        if feature_scores[idx] > 1e-6:
                            top_features.append(feature_names[idx])
                while len(top_features) < n_top:
                    top_features.append("")
                top_features_list.append(top_features[:n_top])
            else:
                fallback_features = feature_names[:n_top] + [
                    "" for _ in range(max(0, n_top - len(feature_names)))
                ]
                top_features_list.append(fallback_features[:n_top])
        return top_features_list


class DataProcessor:
    """
    Data preprocessing and handling class.
    """

    @staticmethod
    def load_and_preprocess(file_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load and preprocess the CSV data.
        """
        df = pd.read_csv(file_path)
        feature_names = [
            col for col in df.columns
            if col.lower() not in ['time', 'timestamp', 'date']
        ]
        for col in feature_names:
            df[col] = df[col].fillna(method='ffill') \
                             .fillna(method='bfill').fillna(0)
        df[feature_names] = \
            df[feature_names].replace([np.inf, -np.inf], np.nan)
        for col in feature_names:
            df[col] = df[col].fillna(df[col].median())
        return df, feature_names

    @staticmethod
    def split_normal_period(
        df: pd.DataFrame, feature_names: List[str], normal_hours: int = 120
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into normal training period and full analysis period.
        """
        normal_end = min(normal_hours, len(df))
        normal_data = df[feature_names].iloc[:normal_end].values
        full_data = df[feature_names].values
        return normal_data, full_data


def main(input_csv_path: str, output_csv_path: str) -> pd.DataFrame:
    """
    Main function to process anomaly detection.
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError(
            "TensorFlow is required. Install with: pip install tensorflow"
        )

    print("Loading and preprocessing data...")
    processor = DataProcessor()
    df, feature_names = processor.load_and_preprocess(input_csv_path)
    normal_data, full_data = processor.split_normal_period(
        df, feature_names, normal_hours=120
    )
    print(f"Data shape: {full_data.shape}")
    print(f"Normal period shape: {normal_data.shape}")
    print(f"Features: {len(feature_names)}")
    print("Training LSTM Autoencoder...")
    detector = LSTMAnomalyDetector(
        sequence_length=min(24, len(normal_data) // 2),
        encoding_dim=32,
        lstm_units=64
    )
    detector.feature_names = feature_names
    detector.fit(normal_data, epochs=30, batch_size=16, validation_split=0.1)
    print("Detecting anomalies...")
    anomaly_scores, _ = detector.predict_anomaly_scores(full_data)
    anomaly_scores = np.array(anomaly_scores).flatten()

    if np.max(anomaly_scores) > np.min(anomaly_scores):
        anomaly_scores = (
            (anomaly_scores - np.min(anomaly_scores))
            / (np.max(anomaly_scores) - np.min(anomaly_scores)) * 100
        )
    else:
        anomaly_scores = np.ones_like(anomaly_scores) * 10

    anomaly_scores += np.random.uniform(0.01, 0.1, size=len(anomaly_scores))
    print("Identifying top contributing features...")
    top_features = detector.get_top_features(full_data, feature_names, n_top=7)

    output_df = df.copy()
    if len(anomaly_scores) != len(output_df):
        if len(anomaly_scores) < len(output_df):
            padded_scores = np.zeros(len(output_df))
            padded_scores[:len(anomaly_scores)] = anomaly_scores
            anomaly_scores = padded_scores
        else:
            anomaly_scores = anomaly_scores[:len(output_df)]

    output_df['Abnormality_score'] = anomaly_scores

    for i in range(7):
        col_name = f'top_feature_{i + 1}'
        if len(top_features) == len(output_df):
            output_df[col_name] = [
                features[i] if i < len(features) else "" for features in top_features
            ]
        else:
            output_df[col_name] = [""] * len(output_df)

    output_df.to_csv(output_csv_path, index=False)
    print(f"Analysis complete. Output saved to: {output_csv_path}")
    training_end = min(120, len(anomaly_scores))
    training_scores = anomaly_scores[:training_end]
    print(
        "Training period anomaly scores - "
        f"Mean: {np.mean(training_scores):.2f}, "
        f"Max: {np.max(training_scores):.2f}"
    )
    return output_df


if __name__ == "__main__":
    input_path = "TEP_Train_Test.csv"
    output_path = "TEP_Train_Test_with_anomalies.csv"

    try:
        result = main(input_path, output_path)
        print("Success! Anomaly detection completed.")
        print(f"Output shape: {result.shape}")
        print("New columns added: Abnormality_score, top_feature_1 through top_feature_7")
    except Exception as error:
        print(f"Error: {error}")
