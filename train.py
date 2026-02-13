import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import joblib

# Load dataset
df = pd.read_csv("projectAIE322.csv")

# Target column
target_col = "GPA"
X = df.drop(columns=[target_col])
y = df[target_col].values
print("X shape:", X.shape, "y shape:", y.shape)
print("Feature columns:", list(X.columns))

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print("Train:", X_train_s.shape, "Test:", X_test_s.shape)

# Train model
model = LinearRegression()
model.fit(X_train_s, y_train)

print("Intercept:", model.intercept_)
print("Coefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(f"  {name}: {coef:.4f}")


#Evaluate with MSE and R²
def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def r2_score_custom(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

y_pred_test = model.predict(X_test_s)

print("Test MSE:", mse(y_test, y_pred_test))
print("Test R2 :", r2_score_custom(y_test, y_pred_test))


joblib.dump(model, "linear_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Saved: linear_model.pkl, scaler.pkl")