from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------
st.markdown("""
<style>
.block-container {
    max-width: 1450px;
    padding: 1.5rem 2rem;
}

.hero {
    background: linear-gradient(110deg, #eaf2ff, #d7e7ff);
    border: 1px solid #cbdcff;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 22px;
}

.hero h1 {
    color: #173778;
    font-size: 2.3rem;
    margin: 0;
}

.hero p {
    color: #344b73;
    font-size: 1.05rem;
    margin-bottom: 0;
}

/* Larger applicant form */
div[data-testid="stForm"] {
    border: 1px solid #d7e1ee;
    border-radius: 15px;
    padding: 20px;
}

div[data-testid="stForm"] [data-testid="stWidgetLabel"] p {
    font-size: 1.12rem !important;
    font-weight: 600 !important;
}

div[data-testid="stForm"] input {
    font-size: 1.1rem !important;
}

div[data-testid="stForm"] [data-baseweb="select"] {
    font-size: 1.1rem !important;
}

/* Prediction panel */
.result-good {
    background: #e6f8ed;
    color: #176534;
    border: 1px solid #a7dfba;
    border-radius: 12px;
    padding: 16px;
    font-size: 1.1rem;
    font-weight: 700;
}

.result-bad {
    background: #fff0f0;
    color: #a51f1f;
    border: 1px solid #ffc4c4;
    border-radius: 12px;
    padding: 16px;
    font-size: 1.1rem;
    font-weight: 700;
}

/* Smaller text on the prediction side */
.prediction-panel h3,
.prediction-panel h4 {
    font-size: 1.05rem !important;
}

.prediction-panel [data-testid="stMetricLabel"] p {
    font-size: 0.78rem !important;
}

.prediction-panel [data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
}

.prediction-panel [data-testid="stCaptionContainer"] {
    font-size: 0.8rem !important;
}

div.stFormSubmitButton > button {
    border-radius: 9px;
    font-size: 1.05rem;
    font-weight: 600;
    min-height: 46px;
}

button[data-baseweb="tab"] {
    font-size: 1rem !important;
}

.footer {
    color: #718096;
    font-size: 0.85rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL AND ENCODERS
# --------------------------------------------------
MODEL_FILES = [
    "xg_boost_credit_model.pkl",
    "XGBoost_credit_model.pkl"
]

model_path = next(
    (Path(name) for name in MODEL_FILES if Path(name).exists()),
    None
)

if model_path is None:
    st.error(
        "Model file not found. Keep your trained model in the same "
        "folder as this Python file."
    )
    st.stop()

ENCODER_NAMES = [
    "Sex",
    "Housing",
    "Saving accounts",
    "Checking account"
]

FEATURES = [
    "Age",
    "Sex",
    "Job",
    "Housing",
    "Saving accounts",
    "Checking account",
    "Credit amount",
    "Duration"
]


@st.cache_resource
def load_artifacts(model_file):
    loaded_model = joblib.load(model_file)

    loaded_encoders = {
        name: joblib.load(f"{name}_encoder.pkl")
        for name in ENCODER_NAMES
    }

    return loaded_model, loaded_encoders


try:
    model, encoders = load_artifacts(str(model_path))
except Exception as error:
    st.error(f"Unable to load model or encoder files: {error}")
    st.stop()


def get_options(name, fallback):
    return [
        str(value)
        for value in encoders[name].classes_.tolist()
    ] or fallback


def get_index(options, preferred):
    return options.index(preferred) if preferred in options else 0


sex_options = get_options("Sex", ["female", "male"])
housing_options = get_options(
    "Housing", ["free", "own", "rent"]
)
saving_options = get_options(
    "Saving accounts", ["little", "moderate", "rich", "quite rich"]
)
checking_options = get_options(
    "Checking account", ["little", "moderate", "rich"]
)

# --------------------------------------------------
# PREDICTION FUNCTIONS
# --------------------------------------------------
def encode_applicant(applicant):
    row = {
        "Age": applicant["Age"],
        "Sex": int(
            encoders["Sex"].transform([applicant["Sex"]])[0]
        ),
        "Job": applicant["Job"],
        "Housing": int(
            encoders["Housing"].transform([applicant["Housing"]])[0]
        ),
        "Saving accounts": int(
            encoders["Saving accounts"].transform(
                [applicant["Saving accounts"]]
            )[0]
        ),
        "Checking account": int(
            encoders["Checking account"].transform(
                [applicant["Checking account"]]
            )[0]
        ),
        "Credit amount": applicant["Credit amount"],
        "Duration": applicant["Duration"]
    }

    return pd.DataFrame(
        [[row[column] for column in FEATURES]],
        columns=FEATURES
    )


def predict_risk(applicant):
    input_df = encode_applicant(applicant)
    prediction = int(model.predict(input_df)[0])

    # Based on your target encoding:
    # 1 = GOOD, 0 = BAD
    result = "GOOD" if prediction == 1 else "BAD"

    scores = {}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]

        for label, probability in zip(
            model.classes_, probabilities
        ):
            name = "GOOD" if int(label) == 1 else "BAD"
            scores[name] = float(probability)

    return prediction, result, scores


def show_prediction(applicant, report_key):
    prediction, result, scores = predict_risk(applicant)

    st.markdown("#### Prediction Result")

    if result == "GOOD":
        st.markdown(
            '<div class="result-good">'
            '✓ Predicted Credit Risk: GOOD'
            '</div>',
            unsafe_allow_html=True
        )
        st.caption("The model classified this application as lower risk.")
    else:
        st.markdown(
            '<div class="result-bad">'
            '! Predicted Credit Risk: BAD'
            '</div>',
            unsafe_allow_html=True
        )
        st.caption("The model classified this application as higher risk.")

    st.write("")

    st.markdown('<div class="prediction-panel">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    col1.metric("Predicted class", result)
    col2.metric("Raw model output", prediction)

    if scores:
        st.markdown("#### Model Output Scores")

        score1, score2 = st.columns(2)

        score1.metric(
            "GOOD score",
            f"{scores.get('GOOD', 0) * 100:.1f}%"
        )

        score2.metric(
            "BAD score",
            f"{scores.get('BAD', 0) * 100:.1f}%"
        )

        st.caption(
            "Model scores do not guarantee loan repayment."
        )

    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("View Applicant Details"):
        st.dataframe(
            pd.DataFrame([applicant]),
            use_container_width=True,
            hide_index=True
        )

    report = pd.DataFrame([{
        **applicant,
        "Predicted Risk": result,
        "Raw Prediction": prediction,
        "GOOD Score": scores.get("GOOD"),
        "BAD Score": scores.get("BAD"),
        "Generated At": datetime.now().isoformat(
            timespec="seconds"
        )
    }])

    st.download_button(
        "Download Prediction Report (CSV)",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="credit_risk_prediction.csv",
        mime="text/csv",
        key=report_key
    )


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>💳 Credit Risk Scoring</h1>
    <p>
        Machine learning based credit risk prediction.
        Enter applicant details, predict risk, and compare applicants.
    </p>
</div>
""", unsafe_allow_html=True)

# Only two sections. About Model has been removed.
single_tab, compare_tab = st.tabs([
    "👤 Single Applicant",
    "⚖️ Compare Applicants"
])

# --------------------------------------------------
# SINGLE APPLICANT
# --------------------------------------------------
with single_tab:
    st.subheader("Applicant Details")
    st.write("Enter the applicant's information to predict credit risk.")

    left, right = st.columns(
        [1.15, 0.85],
        gap="large"
    )

    with left:
        with st.form("single_applicant_form"):
            st.markdown("### Applicant Information")

            col_a, col_b = st.columns(2, gap="large")

            with col_a:
                age = st.number_input(
                    "Age",
                    min_value=18,
                    max_value=80,
                    value=30,
                    step=1
                )

                sex = st.selectbox(
                    "Sex",
                    sex_options,
                    index=get_index(sex_options, "male")
                )

                job = st.selectbox(
                    "Job Level (0 to 3)",
                    [0, 1, 2, 3],
                    index=1
                )

                housing = st.selectbox(
                    "Housing",
                    housing_options,
                    index=get_index(housing_options, "own")
                )

            with col_b:
                saving = st.selectbox(
                    "Saving Accounts",
                    saving_options,
                    index=get_index(saving_options, "moderate")
                )

                checking = st.selectbox(
                    "Checking Account",
                    checking_options,
                    index=get_index(checking_options, "moderate")
                )

                credit = st.number_input(
                    "Credit Amount",
                    min_value=0,
                    max_value=1000000,
                    value=1000,
                    step=100
                )

                duration = st.number_input(
                    "Duration (months)",
                    min_value=1,
                    max_value=120,
                    value=12,
                    step=1
                )

            submitted = st.form_submit_button(
                "📊 Predict Credit Risk",
                type="primary",
                use_container_width=True
            )

    applicant = {
        "Age": int(age),
        "Sex": sex,
        "Job": int(job),
        "Housing": housing,
        "Saving accounts": saving,
        "Checking account": checking,
        "Credit amount": float(credit),
        "Duration": int(duration)
    }

    with right:
        st.markdown('<div class="prediction-panel">', unsafe_allow_html=True)

        if submitted:
            try:
                show_prediction(applicant, "single_report")
            except Exception as error:
                st.error(f"Prediction failed: {error}")
        else:
            st.markdown("### Prediction Result")
            st.info(
                "Enter applicant details on the left and click "
                "Predict Credit Risk."
            )

        st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------
# COMPARE TWO APPLICANTS
# --------------------------------------------------
with compare_tab:
    st.subheader("Compare Two Applicants")
    st.write("Enter information for both applicants and compare their predictions.")

    with st.form("compare_form"):
        applicant_a_col, applicant_b_col = st.columns(2, gap="large")

        with applicant_a_col:
            st.markdown("### Applicant A")

            age_a = st.number_input(
                "Age A", 18, 80, 30, key="age_a"
            )

            sex_a = st.selectbox(
                "Sex A", sex_options,
                index=get_index(sex_options, "male"),
                key="sex_a"
            )

            job_a = st.selectbox(
                "Job Level A", [0, 1, 2, 3],
                index=1, key="job_a"
            )

            housing_a = st.selectbox(
                "Housing A", housing_options,
                index=get_index(housing_options, "own"),
                key="housing_a"
            )

            saving_a = st.selectbox(
                "Saving Accounts A", saving_options,
                index=get_index(saving_options, "moderate"),
                key="saving_a"
            )

            checking_a = st.selectbox(
                "Checking Account A", checking_options,
                index=get_index(checking_options, "moderate"),
                key="checking_a"
            )

            credit_a = st.number_input(
                "Credit Amount A", 0, 1000000, 1000, 100,
                key="credit_a"
            )

            duration_a = st.number_input(
                "Duration A (months)", 1, 120, 12,
                key="duration_a"
            )

        with applicant_b_col:
            st.markdown("### Applicant B")

            age_b = st.number_input(
                "Age B", 18, 80, 40, key="age_b"
            )

            sex_b = st.selectbox(
                "Sex B", sex_options,
                index=get_index(sex_options, "female"),
                key="sex_b"
            )

            job_b = st.selectbox(
                "Job Level B", [0, 1, 2, 3],
                index=2, key="job_b"
            )

            housing_b = st.selectbox(
                "Housing B", housing_options,
                index=get_index(housing_options, "rent"),
                key="housing_b"
            )

            saving_b = st.selectbox(
                "Saving Accounts B", saving_options,
                index=get_index(saving_options, "little"),
                key="saving_b"
            )

            checking_b = st.selectbox(
                "Checking Account B", checking_options,
                index=get_index(checking_options, "little"),
                key="checking_b"
            )

            credit_b = st.number_input(
                "Credit Amount B", 0, 1000000, 5000, 100,
                key="credit_b"
            )

            duration_b = st.number_input(
                "Duration B (months)", 1, 120, 36,
                key="duration_b"
            )

        compare = st.form_submit_button(
            "⚖️ Compare Applicants",
            type="primary",
            use_container_width=True
        )

    if compare:
        applicant_a = {
            "Age": age_a,
            "Sex": sex_a,
            "Job": job_a,
            "Housing": housing_a,
            "Saving accounts": saving_a,
            "Checking account": checking_a,
            "Credit amount": credit_a,
            "Duration": duration_a
        }

        applicant_b = {
            "Age": age_b,
            "Sex": sex_b,
            "Job": job_b,
            "Housing": housing_b,
            "Saving accounts": saving_b,
            "Checking account": checking_b,
            "Credit amount": credit_b,
            "Duration": duration_b
        }

        try:
            pred_a, result_a, scores_a = predict_risk(applicant_a)
            pred_b, result_b, scores_b = predict_risk(applicant_b)

            col_a, col_b = st.columns(2, gap="large")

            with col_a:
                st.markdown("### Applicant A Result")

                if result_a == "GOOD":
                    st.success("Predicted Risk: GOOD")
                else:
                    st.error("Predicted Risk: BAD")

                st.metric("Raw Prediction", pred_a)

                if scores_a:
                    st.metric(
                        "GOOD Score",
                        f"{scores_a.get('GOOD', 0) * 100:.1f}%"
                    )

            with col_b:
                st.markdown("### Applicant B Result")

                if result_b == "GOOD":
                    st.success("Predicted Risk: GOOD")
                else:
                    st.error("Predicted Risk: BAD")

                st.metric("Raw Prediction", pred_b)

                if scores_b:
                    st.metric(
                        "GOOD Score",
                        f"{scores_b.get('GOOD', 0) * 100:.1f}%"
                    )

            comparison = pd.DataFrame([
                {
                    "Applicant": "A",
                    **applicant_a,
                    "Predicted Risk": result_a,
                    "GOOD Score": scores_a.get("GOOD"),
                    "BAD Score": scores_a.get("BAD")
                },
                {
                    "Applicant": "B",
                    **applicant_b,
                    "Predicted Risk": result_b,
                    "GOOD Score": scores_b.get("GOOD"),
                    "BAD Score": scores_b.get("BAD")
                }
            ])

            st.markdown("### Comparison Report")

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Comparison Report (CSV)",
                data=comparison.to_csv(index=False).encode("utf-8"),
                file_name="credit_risk_comparison.csv",
                mime="text/csv"
            )

        except Exception as error:
            st.error(f"Comparison failed: {error}")

st.divider()

st.markdown(
    '<p class="footer">Credit Risk Scoring Demo · Educational use only</p>',
    unsafe_allow_html=True
)