import pandas as pd

def test_dataset_exists():
    df = pd.read_csv("data/cleaned/heart_cleaned.csv")
    assert not df.empty

def test_target_column():
    df = pd.read_csv("data/cleaned/heart_cleaned.csv")
    assert "target" in df.columns
