import joblib

def test_model_file_exists():
    model = joblib.load("models/heart_disease_model.pkl")
    assert model is not None
