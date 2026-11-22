"""Streamlit interface for the Student Performance Prediction project."""

import os
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from main import StudentMLPipeline

try:
    import psycopg2
except ImportError:
    psycopg2 = None

BASE = Path(__file__).resolve().parent
DATA_PATH = str(BASE / "data" / "student-mat.csv")
MODEL_PATH = str(BASE / "models" / "student_model.pkl")

st.set_page_config(page_title="Student Performance AI", page_icon="🎓", layout="wide")
st.markdown(
    """<style>
    .block-container {padding-top: 2rem;}
    .dashboard-title {font-size: 34px; font-weight: 800; color: #243b53;}
    .dashboard-subtitle {color: #627d98; font-size: 16px;}
    .result-card {background: #e8f5e9; padding: 24px; border-radius: 14px;
                  text-align: center; color: #14532d;}
    .result-number {font-size: 42px; font-weight: 800;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    pipeline = StudentMLPipeline(data_path=DATA_PATH, model_path=MODEL_PATH)
    if os.path.exists(MODEL_PATH):
        pipeline.load_model()
    else:
        data = pipeline.load_data()
        pipeline.preprocess_data(data)
        pipeline.train_model()
        pipeline.save_model()
    return pipeline


def get_db_connection():
    if psycopg2 is None:
        raise RuntimeError("Install psycopg2-binary to use prediction history.")
    try:
        url = st.secrets["postgres"]["url"]
    except (KeyError, FileNotFoundError) as exc:
        raise RuntimeError(
            "Missing PostgreSQL URL in .streamlit/secrets.toml or Streamlit Cloud secrets."
        ) from exc
    return psycopg2.connect(url, sslmode="require", connect_timeout=10)


def save_prediction(record):
    sql = """INSERT INTO predictions
        (studytime, failures, absences, g1, g2, schoolsup, famsup, internet, predicted_grade)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    values = (
        record["studytime"], record["failures"], record["absences"],
        record["G1"], record["G2"], record["schoolsup"],
        record["famsup"], record["internet"], record["predicted_grade"],
    )
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, values)


def get_prediction_history():
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM predictions ORDER BY prediction_date DESC")
            rows = cursor.fetchall()
            columns = [column[0] for column in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def clear_prediction_history():
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM predictions")


def generate_prediction_pdf(student_data, prediction):
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40,
    )
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Student Performance Prediction Report", styles["Title"]),
        Spacer(1, 16),
        Paragraph(
            f"Report Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            styles["Normal"],
        ),
        Spacer(1, 20),
        Paragraph("Academic Information", styles["Heading2"]),
    ]
    report_data = [
        ["Parameter", "Value"],
        ["Weekly Study Time (category 1-4)", str(student_data["studytime"])],
        ["Previous Failures", str(student_data["failures"])],
        ["Absences", str(student_data["absences"])],
        ["First Period Grade", str(student_data["G1"])],
        ["Second Period Grade", str(student_data["G2"])],
        ["School Support", str(student_data["schoolsup"])],
        ["Family Support", str(student_data["famsup"])],
        ["Internet Access", str(student_data["internet"])],
    ]
    table = Table(report_data, colWidths=[240, 240])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243b53")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, 0), 12),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 1), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 8),
    ]))
    elements.extend([
        table,
        Spacer(1, 24),
        Paragraph("Predicted Final Grade", styles["Heading2"]),
        Paragraph(f"<b>{prediction:.2f} / 20</b>", styles["Heading1"]),
        Spacer(1, 16),
        Paragraph(
            "This score is an estimate generated by a Random Forest Regression "
            "model using the provided academic information.", styles["Normal"],
        ),
        Spacer(1, 12),
        Paragraph(
            "Disclaimer: This prediction is for educational demonstration only "
            "and should not be used as a definitive assessment of student ability.",
            styles["Normal"],
        ),
    ])
    document.build(elements)
    return buffer.getvalue()


st.sidebar.title("🎓 Student AI")
st.sidebar.write("Machine Learning Dashboard")
page = st.sidebar.radio(
    "Navigation", ["Dashboard", "Model Evaluation", "Prediction Analytics", "About Project"]
)
st.sidebar.divider()
st.sidebar.success("Model: Random Forest")

if page == "Dashboard":
    st.markdown(
        '<div class="dashboard-title">Student Performance Prediction</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="dashboard-subtitle">AI-powered student academic performance estimation</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    try:
        pipeline = load_model()
    except Exception as error:
        st.error(f"Model loading failed: {error}")
        st.stop()

    st.subheader("📋 Student Information")
    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            studytime = st.selectbox(
                "Weekly Study Time", [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Less than 2 hours", 2: "2–5 hours",
                    3: "5–10 hours", 4: "More than 10 hours",
                }[x],
            )
            failures = st.number_input("Previous Failures", min_value=0, max_value=4, value=0)
            absences = st.number_input("Number of Absences", min_value=0, max_value=100, value=2)
            schoolsup = st.selectbox("Extra School Support", ["no", "yes"])
        with right:
            g1 = st.slider("First Period Grade (G1)", 0, 20, 15)
            g2 = st.slider("Second Period Grade (G2)", 0, 20, 16)
            famsup = st.selectbox("Family Support", ["yes", "no"])
            internet = st.selectbox("Internet Access", ["yes", "no"])
        submitted = st.form_submit_button(
            "🚀 Predict Student Performance", type="primary", use_container_width=True
        )

    if submitted:
        student_data = {
            "studytime": studytime, "failures": failures, "absences": absences,
            "G1": g1, "G2": g2, "schoolsup": schoolsup,
            "famsup": famsup, "internet": internet,
        }
        try:
            prediction = pipeline.predict_student(student_data)
            pdf_bytes = generate_prediction_pdf(student_data, prediction)
            st.session_state["last_prediction"] = {
                "prediction": prediction, "G1": g1, "G2": g2,
                "pdf_bytes": pdf_bytes,
            }
        except Exception as error:
            st.error(f"Prediction or PDF generation failed: {error}")
        else:
            try:
                save_prediction({**student_data, "predicted_grade": round(prediction, 2)})
                st.success("Prediction saved to PostgreSQL.")
            except Exception as error:
                st.warning(f"Prediction calculated, but cloud history was not saved: {error}")

    if "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]
        prediction = result["prediction"]
        st.subheader("📊 Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="result-card"><h3>Predicted Final Grade</h3>'
                f'<div class="result-number">{prediction:.2f}/20</div>'
                '<p>Estimated academic score</p></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.metric("Predicted Grade", f"{prediction:.2f}/20")
            st.progress(min(max(prediction / 20, 0.0), 1.0))
            st.write(f"Estimated score: {prediction / 20 * 100:.1f}%")

        chart_data = pd.DataFrame({
            "Category": ["First Period", "Second Period", "Predicted Final"],
            "Grade": [result["G1"], result["G2"], prediction],
        }).set_index("Category")
        st.subheader("📈 Grade Visualization")
        st.bar_chart(chart_data)
        st.info("This is an educational estimate, not a definitive assessment of student ability.")
        st.download_button(
            "📄 Download Prediction PDF Report", data=result["pdf_bytes"],
            file_name="student_prediction_report.pdf", mime="application/pdf",
            use_container_width=True,
        )

    st.divider()
    st.subheader("🗄️ Cloud Prediction History")
    try:
        history = get_prediction_history()
    except Exception as error:
        st.warning(f"Prediction history unavailable: {error}")
    else:
        if history.empty:
            st.info("No prediction records found yet.")
        else:
            st.metric("Total Predictions", len(history))
            st.dataframe(history, use_container_width=True)
            st.download_button(
                "📥 Download CSV Report", history.to_csv(index=False).encode("utf-8"),
                file_name="student_prediction_history.csv", mime="text/csv",
            )
            with st.expander("Database maintenance"):
                st.warning("This deletes the shared prediction history for every user of this app.")
                confirm = st.checkbox("I understand this will delete all stored prediction records.")
                if st.button("🗑️ Clear All Prediction Records", disabled=not confirm):
                    try:
                        clear_prediction_history()
                        st.success("Prediction records cleared.")
                        st.rerun()
                    except Exception as error:
                        st.error(f"Could not clear records: {error}")

elif page == "Model Evaluation":
    st.title("📊 Model Evaluation Dashboard")
    st.write("Evaluate the saved Random Forest Regression model.")
    try:
        pipeline = load_model()
        results = pipeline.evaluate_saved_model()
    except Exception as error:
        st.error(f"Evaluation failed: {error}")
    else:
        a, b, c = st.columns(3)
        a.metric("MAE", f"{results['mae']:.2f}")
        b.metric("RMSE", f"{results['rmse']:.2f}")
        c.metric("R² Score", f"{results['r2']:.3f}")
        st.caption("Lower MAE and RMSE indicate smaller errors; R² describes explained variance.")
        st.divider()
        st.subheader("Actual vs Predicted Grades")
        st.scatter_chart(results["comparison"], x="Actual Grade", y="Predicted Grade")
        st.subheader("Top 10 Feature Importances")
        importance = results["importance"].head(10)
        st.bar_chart(importance.set_index("Feature")["Importance"])
        st.subheader("Evaluation Data")
        st.dataframe(results["comparison"], use_container_width=True)
        st.subheader("Feature Importance Data")
        st.dataframe(results["importance"], use_container_width=True)

elif page == "Prediction Analytics":
    st.title("📊 Prediction Analytics Dashboard")
    st.write("Analyze student performance predictions stored in Supabase PostgreSQL.")
    try:
        history_df = get_prediction_history()
    except Exception as error:
        st.error(f"Analytics unavailable: {error}")
    else:
        if history_df.empty:
            st.info("No prediction records found. Create predictions from the Dashboard first.")
        else:
            history_df["predicted_grade"] = pd.to_numeric(
                history_df["predicted_grade"], errors="coerce"
            )
            history_df["prediction_date"] = pd.to_datetime(
                history_df["prediction_date"], errors="coerce", utc=True
            )
            valid_df = history_df.dropna(subset=["predicted_grade"]).copy()
            if valid_df.empty:
                st.warning("No valid grades available.")
            else:
                st.subheader("📈 Performance Overview")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Predictions", len(valid_df))
                col2.metric("Average Grade", f"{valid_df['predicted_grade'].mean():.2f}/20")
                col3.metric("Highest Grade", f"{valid_df['predicted_grade'].max():.2f}/20")
                col4.metric("Lowest Grade", f"{valid_df['predicted_grade'].min():.2f}/20")
                st.divider()

                st.subheader("📊 Predicted Grade Distribution")
                grade_counts = (
                    valid_df["predicted_grade"].round().value_counts().sort_index()
                    .rename_axis("Grade").reset_index(name="Count").set_index("Grade")
                )
                st.bar_chart(grade_counts)
                st.divider()

                st.subheader("📚 Study Time vs Predicted Grade")
                study_data = (
                    valid_df.groupby("studytime")["predicted_grade"]
                    .mean().reset_index().set_index("studytime")
                )
                st.bar_chart(study_data)
                st.caption(
                    "These averages describe saved predictions; they do not establish cause and effect."
                )
                st.divider()

                st.subheader("📅 Prediction Trends")
                trend_df = valid_df.dropna(subset=["prediction_date"]).sort_values(
                    "prediction_date"
                )
                if not trend_df.empty:
                    st.line_chart(trend_df.set_index("prediction_date")["predicted_grade"])
                else:
                    st.info("No valid prediction dates available.")
                st.divider()
                st.subheader("📋 Analytics Data")
                st.dataframe(valid_df, use_container_width=True)

elif page == "About Project":
    st.title("ℹ️ About This Project")
    st.write(
        "Student Performance Prediction is an end-to-end machine learning project "
        "that estimates a final grade from academic and support-related inputs."
    )
    st.subheader("Technologies Used")
    st.write(
        "Python · Pandas · NumPy · Scikit-learn · Random Forest Regression · "
        "Streamlit · Joblib · PostgreSQL · ReportLab"
    )
    st.divider()
    st.subheader("Machine Learning Workflow")
    for index, step in enumerate([
        "Data Collection", "Data Preprocessing", "Feature Engineering",
        "Model Training", "Model Evaluation", "Model Saving", "Grade Prediction",
    ], start=1):
        st.write(f"{index}. {step}")
    st.caption("Student Performance Prediction | End-to-End ML Pipeline Project")
