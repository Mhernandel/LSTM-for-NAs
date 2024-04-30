import pandas as pd
from prepareData import PrepareData
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_and_save_model(filepath, model_path):
    processor = PrepareData(filepath)
    processor.load_data()
    data_scaled = processor.scale_data()
    X_train = processor.reshape_data_for_lstm()
    num_features = data_scaled.shape[1]

    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(num_features)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, data_scaled, epochs=50, batch_size=10)

    model.save(model_path)

