import pandas as pd

def test_dataset_exists():
    df = pd.read_csv("data/cleaned/heart_clean.csv")
    assert not df.empty

def test_target_column():
    df = pd.read_csv("data/cleaned/heart_clean.csv")
    assert "target" in df.columns
