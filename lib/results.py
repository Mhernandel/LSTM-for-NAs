import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prepareData import PrepareData
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def load_and_predict(filepath, model_path):
    model = load_model(model_path)

    processor = PrepareData(filepath)
    processor.load_data()
    processor.scale_data()
    X_train = processor.reshape_data_for_lstm()
    

    predicted = model.predict(X_train)
    predicted_values = processor.inverse_transform(predicted)
    

    return processor.data, predicted_values

def plot_results(data, predicted_values):
    fig, ax = plt.subplots(figsize=(15, 5))
    for index, country in enumerate(data.index[:2]):  
        ax.plot(data.columns, data.iloc[index], label=f'Actual - {country}')
        ax.plot(data.columns, predicted_values[index], label=f'Predicted - {country}', linestyle='--')
    ax.set_title('Prediction vs Actual')
    ax.set_xlabel('Year')
    ax.set_ylabel('Values')
    ax.legend()

    pngImage = BytesIO()
    fig.savefig(pngImage, format='png')
    pngImage.seek(0)  
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    plt.close(fig)

    return pngImageB64String

def save_predictions_to_csv(data, predicted_values, save_path):
    predicted_df = pd.DataFrame(predicted_values, index=data.index, columns=data.columns)
    combined_data = data.combine_first(predicted_df)
    combined_data.to_csv(save_path)

