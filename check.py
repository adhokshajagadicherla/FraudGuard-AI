import joblib, json, pandas as pd

lr = joblib.load("models/logistic_regression.pkl")
rf = joblib.load("models/random_forest.pkl")
sc = joblib.load("models/scaler.pkl")
mt = json.load(open("models/metrics.json"))
df = pd.read_csv("data/sample_data.csv")

print(f"[OK] LR loaded: {type(lr).__name__}")
print(f"[OK] RF loaded: {type(rf).__name__}")
print(f"[OK] Scaler: {type(sc).__name__}")
print(f"[OK] Metrics keys: {list(mt.keys())}")
print(f"[OK] Sample data: {len(df)} rows, {len(df.columns)} cols")
print(f"[OK] Fraud in sample: {df['Class'].sum()} rows")

sample = df.drop(columns=["Class"]).iloc[[0]].values
scaled = sc.transform(sample)
lr_p = lr.predict_proba(scaled)[0][1]
rf_p = rf.predict_proba(scaled)[0][1]
print(f"[OK] LR prediction: {lr_p:.4f}  RF prediction: {rf_p:.4f}")
print("[ALL CHECKS PASSED]")
