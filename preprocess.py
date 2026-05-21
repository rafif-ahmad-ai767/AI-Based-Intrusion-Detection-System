import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)

    df = df.dropna()
    df = df.drop_duplicates()

    label = LabelEncoder()
    df['Label'] = label.fit_transform(df['Label'])

    X = df.drop('Label', axis=1)
    y = df['Label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, label
