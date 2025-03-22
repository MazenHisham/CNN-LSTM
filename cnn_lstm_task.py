import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, Dropout

def load_stock_data(csv_file, stock_column="AMZN"):
    df = pd.read_csv(csv_file)

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    if stock_column not in df.columns:
        raise KeyError(f"'{stock_column}' column not found in the dataset.")

    return df[[stock_column]].rename(columns={stock_column: "Close"})

def create_financial_charts(data, output_folder, window_size=30):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(len(data) - window_size):
        plt.figure(figsize=(5, 3))
        plt.plot(data.index[i:i+window_size], data['Close'][i:i+window_size], color='blue')
        plt.axis('off')
        plt.savefig(f"{output_folder}/chart_{i}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

def load_images(image_folder, img_size=(64, 64)):
    images = []
    for img_name in sorted(os.listdir(image_folder)):
        img = cv2.imread(os.path.join(image_folder, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size) / 255.0  # Normalize
        images.append(img)
    return np.array(images)

df = load_stock_data('portfolio_data.csv', stock_column="AMZN")

create_financial_charts(df, 'charts')

X = load_images('charts')
X = np.expand_dims(X, axis=-1)  # Add channel dimension
X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2], X.shape[3])  # Reshape for LSTM

y = df['Close'][30:].values  # Offset labels by window size

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential([
    TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(1, 64, 64, 1))),
    TimeDistributed(MaxPooling2D(2, 2)),
    TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')),
    TimeDistributed(MaxPooling2D(2, 2)),
    TimeDistributed(Flatten()),

    LSTM(50, activation='relu', return_sequences=False),
    Dropout(0.2),

    Dense(50, activation='relu'),
    Dense(1)  # Output: next day's closing price
])

model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_test, y_test))

predictions = model.predict(X_test)

plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()
