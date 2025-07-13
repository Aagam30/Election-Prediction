
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Generate synthetic training data
np.random.seed(42)
n_samples = 5000

# Create realistic election data
data = {
    'age': np.random.normal(50, 15, n_samples),
    'income': np.random.normal(65000, 25000, n_samples),
    'education': np.random.normal(14, 3, n_samples),
    'sentiment': np.random.beta(2, 2, n_samples),  # More realistic sentiment distribution
    'poll': np.random.beta(3, 3, n_samples)  # More realistic polling distribution
}

# Create more sophisticated target variable
def calculate_win_probability(age, income, education, sentiment, poll):
    # Weighted formula based on political science research
    prob = (
        0.35 * poll +  # Polling is strongest predictor
        0.25 * sentiment +  # Sentiment matters significantly
        0.15 * (education / 20) +  # Education normalized
        0.15 * (1 - abs(age - 55) / 55) +  # Age sweet spot around 55
        0.10 * (income / 150000)  # Income influence (capped)
    )
    # Add some noise for realism
    prob += np.random.normal(0, 0.1)
    return np.clip(prob, 0, 1)

# Generate target variable
win_probs = [calculate_win_probability(data['age'][i], data['income'][i], 
                                     data['education'][i], data['sentiment'][i], 
                                     data['poll'][i]) for i in range(n_samples)]

# Convert to binary classification (win/lose)
targets = [1 if prob > 0.5 else 0 for prob in win_probs]

# Create DataFrame
df = pd.DataFrame(data)
df['target'] = targets

# Prepare features
X = df[['age', 'income', 'education', 'sentiment', 'poll']]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model (more accurate than Decision Tree)
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_names = ['age', 'income', 'education', 'sentiment', 'poll']
importances = model.feature_importances_
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.3f}")

# Save the model and scaler
with open('election_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nModel and scaler saved successfully!")
