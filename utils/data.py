# utils/data.py
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    # adjust to your dataset columns (UCI heart failure example shown)
    'age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction',
    'high_blood_pressure','platelets','serum_creatinine','serum_sodium',
    'sex','smoking','time'
]
TARGET = 'DEATH_EVENT'

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def split_xy(df):
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    return X, y

def train_valid_test_split(X, y, test_size=0.2, valid_size=0.2, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=valid_size, random_state=random_state, stratify=y_temp
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test
