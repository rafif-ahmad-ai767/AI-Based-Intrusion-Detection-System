import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocess import load_and_preprocess

X, y, scaler, label = load_and_preprocess("data/CICIDS2017.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))

joblib.dump(model, "models/ids_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label, "models/label_encoder.pkl")
