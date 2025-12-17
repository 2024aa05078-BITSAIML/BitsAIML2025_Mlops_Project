import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from preprocess import build_preprocessing_pipeline

# Load data
data = pd.read_csv("data/cleaned/heart_clean.csv")
X = data.drop("target", axis=1)
y = data["target"]

# Build pipeline
preprocessor = build_preprocessing_pipeline(X)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

final_pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", model)
])

# Train
final_pipeline.fit(X, y)

# Save model
joblib.dump(final_pipeline, "models/heart_disease_model.pkl")

print("Final model saved successfully!")
