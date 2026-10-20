import hmac
import json
import hashlib
import secrets
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(page_title="FraudGuard AI | Credit Card Fraud Detection", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.stApp {background:#0a0e1a;color:#e7edf8}
section[data-testid="stSidebar"] {background:#0d1424;border-right:1px solid #24314d}
h1,h2,h3 {color:#eaf2ff}
div[data-testid="stMetric"] {background:#151f35;border:1px solid #263958;padding:16px;border-radius:14px}
.small-muted {color:#91a4c5}
</style>
""", unsafe_allow_html=True)


def find_model(names):
    for name in names:
        path = MODEL_DIR / name
        if path.exists():
            try:
                return joblib.load(path)
            except Exception:
                pass
    return None


def load_models():
    return {
        "Logistic Regression": find_model(["logistic_regression.pkl", "logistic_model.pkl"]),
        "Random Forest": find_model(["random_forest.pkl", "random_forest_model.pkl"]),
        "Scaler": find_model(["scaler.pkl"]),
    }


def load_metrics():
    path = MODEL_DIR / "metrics.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


@st.cache_data(show_spinner=False)
def load_dataset():
    candidates = [
        DATA_DIR / "creditcard.csv", DATA_DIR / "sample_data.csv",
        BASE_DIR / "creditcard.csv", BASE_DIR / "sample_data.csv",
    ]
    for path in candidates:
        if path.exists():
            try:
                return pd.read_csv(path), f"Loaded {path.name}"
            except Exception as exc:
                return None, f"Could not read {path.name}: {exc}"
    return None, "Dataset not found. Put creditcard.csv or sample_data.csv inside the data folder."


def get_target(df):
    for col in ["Class", "class", "label", "target", "is_fraud", "fraud"]:
        if col in df.columns:
            return col
    return None


def _account_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DATA_DIR / "accounts.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
    return db_path


def _hash_password(password, salt_hex=None):
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return digest.hex(), salt.hex()


def _register_user(username, password):
    username = username.strip()
    if len(username) < 3:
        return False, "Username must contain at least 3 characters."
    if len(username) > 40 or any(ch.isspace() for ch in username):
        return False, "Username must be 3–40 characters and contain no spaces."
    if len(password) < 8:
        return False, "Password must contain at least 8 characters."
    password_hash, salt = _hash_password(password)
    try:
        with sqlite3.connect(_account_db()) as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)",
                (username, password_hash, salt),
            )
        return True, "Account created. You can now log in."
    except sqlite3.IntegrityError:
        return False, "That username already exists. Please choose another."


def _verify_user(username, password):
    with sqlite3.connect(_account_db()) as conn:
        row = conn.execute(
            "SELECT password_hash, salt FROM users WHERE username = ? COLLATE NOCASE",
            (username.strip(),),
        ).fetchone()
    if not row:
        return False
    actual_hash, _ = _hash_password(password, row[1])
    return hmac.compare_digest(actual_hash, row[0])


def _verify_admin(username, password):
    # Optional admin account configured in .streamlit/secrets.toml.
    try:
        expected_user = st.secrets.get("APP_USERNAME", "")
        expected_password = st.secrets.get("APP_PASSWORD", "")
    except Exception:
        return False
    return bool(
        expected_user and expected_password
        and hmac.compare_digest(username, expected_user)
        and hmac.compare_digest(password, expected_password)
    )


def login_page():
    st.markdown("<div style='height:45px'></div><h1 style='text-align:center;color:#00D4FF'>🛡️ FraudGuard AI</h1><p style='text-align:center;color:#91a4c5'>Credit Card Fraud Detection System</p>", unsafe_allow_html=True)
    _, center, _ = st.columns([1, 1.2, 1])
    with center:
        login_tab, signup_tab = st.tabs(["Login", "Create Account"])
        with login_tab:
            with st.form("fraudguard_login"):
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Login", use_container_width=True)
            if submitted:
                username = username.strip()
                if _verify_admin(username, password) or _verify_user(username, password):
                    st.session_state["authenticated"] = True
                    st.session_state["logged_in_user"] = username
                    st.rerun()
                else:
                    st.error("Username or password is incorrect.")
        with signup_tab:
            st.write("Create an account to access the FraudGuard AI dashboard.")
            with st.form("fraudguard_signup", clear_on_submit=True):
                new_username = st.text_input("Choose a username", key="signup_username")
                new_password = st.text_input("Create password", type="password", key="signup_password")
                confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm")
                submitted_signup = st.form_submit_button("Create Account", use_container_width=True)
            if submitted_signup:
                if new_password != confirm_password:
                    st.error("The passwords do not match.")
                else:
                    success, message = _register_user(new_username, new_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        st.caption("Educational demo only. Never enter real card numbers, CVVs, or payment information.")


def sidebar():
    with st.sidebar:
        st.markdown("<h1 style='text-align:center;color:#00D4FF'>🛡️ FraudGuard AI</h1><p style='text-align:center;color:#91a4c5'>Credit Card Fraud Detection</p>", unsafe_allow_html=True)
        page = st.radio("Navigate", ["🏠 Dashboard", "📊 Data Explorer", "🤖 Model Performance", "🔮 Fraud Predictor", "ℹ️ About"], label_visibility="collapsed")
        st.divider()
        st.caption("SRM Institute of Science and Technology, Tiruchirappalli")
        st.caption("B.Tech CSE–BDA | III Year")
        if st.session_state.get("logged_in_user"):
            st.caption("Signed in as: " + str(st.session_state["logged_in_user"]))
        if st.button("Log out", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state.pop("logged_in_user", None)
            st.rerun()
    return page


def page_dashboard(df, status, metrics, models):
    st.title("FraudGuard AI")
    st.write("Credit card fraud detection • Machine learning • Transaction risk analysis")
    if df is None:
        st.warning(status)
        st.markdown("[Open the Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")
        st.info("Place the downloaded CSV file at data/creditcard.csv or data/sample_data.csv.")
        if models:
            st.write("Model files detected.")
        return
    target = get_target(df)
    total = len(df)
    fraud = int((df[target] == 1).sum()) if target else 0
    legit = total - fraud if target else 0
    rate = (fraud / total * 100) if target and total else 0
    a, b, c, d = st.columns(4)
    a.metric("Transactions in loaded data", f"{total:,}")
    b.metric("Fraud cases", f"{fraud:,}" if target else "No label column")
    c.metric("Legitimate cases", f"{legit:,}" if target else "—")
    d.metric("Fraud rate", f"{rate:.3f}%" if target else "—")
    if target:
        counts = df[target].map({0: "Legitimate", 1: "Fraud"}).fillna(df[target].astype(str)).value_counts().rename_axis("Type").reset_index(name="Count")
        left, right = st.columns(2)
        with left:
            st.subheader("Transaction class distribution")
            st.plotly_chart(px.bar(counts, x="Type", y="Count", color="Type"), use_container_width=True)
        with right:
            st.subheader("Class share")
            st.plotly_chart(px.pie(counts, names="Type", values="Count", hole=0.55), use_container_width=True)
    st.subheader("Model status")
    loaded = [name for name, model in models.items() if model is not None]
    if loaded:
        st.success("Loaded: " + ", ".join(loaded))
    else:
        st.warning("No trained models found. Run train_model.py before using prediction.")
    if metrics:
        st.subheader("Saved evaluation metrics")
        st.json(metrics)


def page_data_explorer(df, status):
    st.title("📊 Data Explorer")
    if df is None:
        st.warning(status)
        return
    st.write(f"**Rows:** {len(df):,} | **Columns:** {len(df.columns)}")
    st.dataframe(df.head(100), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Missing values")
        missing = df.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        st.dataframe(missing.rename("Missing values") if len(missing) else pd.DataFrame({"Status": ["No missing values found"]}), use_container_width=True)
    with right:
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        if numeric:
            feature = st.selectbox("Feature distribution", numeric)
            st.plotly_chart(px.histogram(df, x=feature, nbins=40), use_container_width=True)
    with st.expander("Summary statistics"):
        st.dataframe(df.describe(include="all").T, use_container_width=True)


def page_performance(df, metrics, models):
    st.title("🤖 Model Performance")
    if metrics:
        st.subheader("Saved evaluation metrics")
        st.json(metrics)
    else:
        st.info("No metrics.json found. Train the models and save their evaluation metrics.")
    if df is not None:
        target = get_target(df)
        if target:
            st.subheader("Class balance")
            st.bar_chart(df[target].value_counts().sort_index())
    st.subheader("Available models")
    for name, model in models.items():
        st.write(f"{'✅' if model is not None else '—'} {name}")


def page_predictor(models):
    st.title("🔮 Fraud Predictor")
    st.warning("Educational demonstration only. Do not use this result to make real financial decisions.")
    available = [name for name in ["Logistic Regression", "Random Forest"] if models.get(name) is not None]
    if not available:
        st.error("No trained classifier found. Check models/ and run train_model.py.")
        return
    model_name = st.selectbox("Choose model", available)
    model = models[model_name]
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    else:
        feature_names = list(feature_names)
    with st.form("predict_transaction"):
        st.caption("Enter feature values from your dataset. Zero defaults are only placeholders.")
        cols = st.columns(3)
        values = {}
        for i, name in enumerate(feature_names):
            with cols[i % 3]:
                values[name] = st.number_input(str(name), value=0.0, format="%.6f", key=f"feature_{name}")
        submitted = st.form_submit_button("Analyze transaction", use_container_width=True)
    if submitted:
        try:
            X = pd.DataFrame([values], columns=feature_names)
            scaler = models.get("Scaler")
            if scaler is not None:
                scale_cols = [c for c in ["Time", "Amount"] if c in X.columns]
                if scale_cols:
                    X.loc[:, scale_cols] = scaler.transform(X[scale_cols])
            prediction = int(model.predict(X)[0])
            if prediction == 1:
                st.error("Model result: Potential fraud")
            else:
                st.success("Model result: Likely legitimate")
            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(X)[0]
                if len(probability) > 1:
                    st.metric("Estimated fraud probability", f"{float(probability[1]):.2%}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.caption("Input features and preprocessing must match the training pipeline exactly.")


def page_about():
    st.title("ℹ️ About FraudGuard AI")
    st.write("A student project demonstrating supervised machine learning for detecting potentially fraudulent transactions in anonymised credit-card data.")
    st.subheader("Methodology")
    st.markdown("""
- **Dataset:** Kaggle Credit Card Fraud Detection dataset.
- **Preprocessing:** Stratified train/test split; fit preprocessing on training data only.
- **Models:** Logistic Regression and Random Forest.
- **Evaluation:** Precision, recall, F1-score, PR-AUC, ROC-AUC, and confusion matrix.
- **Class imbalance:** Accuracy alone is insufficient when fraud is rare.
""")
    st.markdown("[Open the Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")
    st.subheader("Technology stack")
    st.write("Python • Streamlit • Pandas • NumPy • scikit-learn • Plotly • Joblib")
    st.caption("Educational project only. Never enter real payment information.")


def main():
    if not st.session_state.get("authenticated", False):
        login_page()
        return
    page = sidebar()
    df, status = load_dataset()
    metrics = load_metrics()
    models = load_models()
    if page == "🏠 Dashboard":
        page_dashboard(df, status, metrics, models)
    elif page == "📊 Data Explorer":
        page_data_explorer(df, status)
    elif page == "🤖 Model Performance":
        page_performance(df, metrics, models)
    elif page == "🔮 Fraud Predictor":
        page_predictor(models)
    else:
        page_about()


if __name__ == "__main__":
    main()
