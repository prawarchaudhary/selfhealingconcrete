# selfhealingconcrete
A open repo for CNN based code for paper on self healing concrete
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Simulate loading dataset
# Replace this with: df = pd.read_csv('krkCMd_brightness_profiles.csv')
# Data format: 501 brightness values (x1 to x501) + 'crack_width'
num_samples = 19098
np.random.seed(42)
X = np.random.normal(128, 25, size=(num_samples, 501))  # simulate brightness
y = 200 - 3*np.log1p(np.arange(num_samples) % 6 + 1) + np.random.normal(0, 3, size=(num_samples,))  # simulated width

# Convert to DataFrame
df = pd.DataFrame(X, columns=[f'x{i}' for i in range(1, 502)])
df['crack_width'] = y

# Feature matrix and target
X = df.drop(columns='crack_width').values
y = df['crack_width'].values

# Z-score normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for Conv1D: (samples, time steps, features)
X_scaled = X_scaled.reshape(-1, 501, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv1D(32, kernel_size=5, activation='relu', input_shape=(501, 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)  # Regression output
])

model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

# Predict
y_pred = model.predict(X_test).flatten()

# Evaluation metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f'R²: {r2:.4f}')
print(f'RMSE: {rmse:.2f} µm')
print(f'MAE: {mae:.2f} µm')

# Visualization
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Crack Width (µm)')
plt.ylabel('Predicted Crack Width (µm)')
plt.title('Actual vs Predicted Crack Width')
plt.grid(True)
plt.tight_layout()
plt.show()
