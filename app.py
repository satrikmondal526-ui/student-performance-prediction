from datetime import datetime
import os
import streamlit as st
import pandas as pd
from main import StudentMLPipeline

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)
# Initialize Prediction History

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# --------------------------------
# CUSTOM CSS
# --------------------------------

st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.block-container {
    padding-top: 2rem;
}

.dashboard-title {
    font-size: 34px;
    font-weight: 800;
    color: #243b53;
}

.dashboard-subtitle {
    color: #627d98;
    font-size: 16px;
}

.result-card {
    background-color: #e8f5e9;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}

.result-number {
    font-size: 42px;
    font-weight: bold;
    color: #16803c;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# LOAD TRAINED MODEL
# --------------------------------


@st.cache_resource
def load_model():

    pipeline = StudentMLPipeline(
        data_path="data/student-mat.csv",
        model_path="models/student_model.pkl"
    )

    if os.path.exists("models/student_model.pkl"):

        pipeline.load_model()

    else:

        st.info("Training machine learning model...")

        data = pipeline.load_data()

        pipeline.preprocess_data(data)

        pipeline.train_model()

        pipeline.save_model()

    return pipeline



# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.title("🎓 Student AI")

st.sidebar.write("Machine Learning Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Model Evaluation",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.success("Model: Random Forest")


# --------------------------------
# DASHBOARD PAGE
# --------------------------------

if page == "Dashboard":

    st.markdown(
        '<div class="dashboard-title">'
        'Student Performance Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        'AI-powered student academic performance estimation'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # Load model

    try:
        pipeline = load_model()
        st.success("Machine Learning Model Loaded Successfully!")

    except Exception as error:
        st.error(f"Model loading failed: {error}")
        st.stop()

    # Student information

    st.subheader("📋 Student Information")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Academic Information")

        studytime = st.selectbox(
            "Weekly Study Time",
            [1, 2, 3, 4],
            format_func=lambda x: {
                1: "Less than 2 hours",
                2: "2–5 hours",
                3: "5–10 hours",
                4: "More than 10 hours"
            }[x]
        )

        failures = st.number_input(
            "Previous Failures",
            min_value=0,
            max_value=4,
            value=0
        )

        absences = st.number_input(
            "Number of Absences",
            min_value=0,
            max_value=100,
            value=2
        )

    with col2:

        st.markdown("### Previous Grades")

        g1 = st.slider(
            "First Period Grade (G1)",
            0, 20, 15
        )

        g2 = st.slider(
            "Second Period Grade (G2)",
            0, 20, 16
        )

    st.divider()

    st.subheader("🏠 Student Support Information")

    col3, col4, col5 = st.columns(3)

    with col3:
        schoolsup = st.selectbox(
            "Extra School Support",
            ["yes", "no"]
        )

    with col4:
        famsup = st.selectbox(
            "Family Support",
            ["yes", "no"]
        )

    with col5:
        internet = st.selectbox(
            "Internet Access",
            ["yes", "no"]
        )

    st.divider()

    # Prediction button

    if st.button(
        "🚀 Predict Student Performance",
        type="primary",
        use_container_width=True
    ):

        student_data = {

            "studytime": studytime,
            "failures": failures,
            "absences": absences,
            "G1": g1,
            "G2": g2,
            "schoolsup": schoolsup,
            "famsup": famsup,
            "internet": internet

        }

        try:

            prediction = pipeline.predict_student(
                student_data
            )

    # Save Prediction History

            history_record = {

             "Date & Time": datetime.now().strftime(
             "%Y-%m-%d %H:%M:%S"
             ),

            "Study Time": studytime,
            "Failures": failures,
            "Absences": absences,
            "G1": g1,
            "G2": g2,
            "School Support": schoolsup,
            "Family Support": famsup,
            "Internet": internet,
            "Predicted Grade": round(prediction, 2)

            }

            st.session_state.prediction_history.append(
            history_record
           )          

            st.divider()

            st.subheader("📊 Prediction Results")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.markdown(
                    f"""
                    <div class="result-card">
                        <h3>Predicted Final Grade</h3>
                        <div class="result-number">
                            {prediction:.2f}/20
                        </div>
                        <p>Estimated Academic Score</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with result_col2:

                st.metric(
                    "Predicted Grade",
                    f"{prediction:.2f}/20"
                )

                st.progress(
                    prediction / 20
                )

                st.write(
                    f"Estimated score: {prediction / 20 * 100:.1f}%"
                )

            # Grade visualization

            st.subheader("📈 Grade Visualization")

            chart_data = pd.DataFrame({
                "Category": [
                    "First Period",
                    "Second Period",
                    "Predicted Final"
                ],
                "Grade": [
                    g1,
                    g2,
                    prediction
                ]
            })

            st.bar_chart(
                chart_data.set_index("Category")
            )

            st.info(
                "This prediction is an educational estimate "
                "and should not be treated as a definitive "
                "assessment of student ability."
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# --------------------------------
# PREDICTION HISTORY
# --------------------------------

    st.divider()

    st.subheader("📜 Prediction History")

    if st.session_state.prediction_history:

        history_df = pd.DataFrame(
            st.session_state.prediction_history
        )

        # Display History

        st.dataframe(
            history_df,
            use_container_width=True
        )

        # Download CSV Report

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Prediction Report",
            data=csv_data,
            file_name="student_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Clear History

        if st.button(
            "🗑️ Clear Prediction History"
        ):

            st.session_state.prediction_history = []

            st.rerun()

    else:

        st.info(
            "No prediction history available. "
            "Make your first student prediction!"
        )


# --------------------------------
# ABOUT PROJECT PAGE
# --------------------------------

elif page == "About Project":

    st.title("ℹ️ About This Project")

    st.write(
        "Student Performance Prediction is an "
        "end-to-end Machine Learning project."
    )

    st.subheader("Technologies Used")

    technologies = [
        "Python",
        "Pandas",
        "NumPy",
        "Scikit-learn",
        "Random Forest Regression",
        "Streamlit",
        "Matplotlib",
        "Joblib"
    ]

    for technology in technologies:
        st.write("✅", technology)

    st.divider()

    st.subheader("Machine Learning Workflow")

    workflow = [
        "Data Collection",
        "Data Preprocessing",
        "Feature Engineering",
        "Model Training",
        "Model Evaluation",
        "Model Saving",
        "Student Grade Prediction"
    ]

    for index, step in enumerate(workflow, 1):
        st.write(f"**{index}.** {step}")

    st.divider()

    st.caption(
        "Student Performance Prediction | "
        "End-to-End ML Pipeline Project"
    )


# --------------------------------
# MODEL EVALUATION PAGE
# --------------------------------

elif page == "Model Evaluation":

    st.title("📊 Model Evaluation Dashboard")

    st.write(
        "Evaluate the performance of the "
        "trained Random Forest Regression model."
    )

    try:

        pipeline = load_model()

        results = pipeline.evaluate_saved_model()

        st.success("Model evaluation completed!")

        # Performance Metrics

        st.subheader("Performance Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "MAE",
                f"{results['mae']:.2f}"
            )

        with col2:
            st.metric(
                "RMSE",
                f"{results['rmse']:.2f}"
            )

        with col3:
            st.metric(
                "R² Score",
                f"{results['r2']:.3f}"
            )

        st.caption(
            "Lower MAE and RMSE generally indicate "
            "smaller prediction errors. R² measures "
            "how well the model explains variation "
            "in the test data."
        )

        st.divider()

        # Actual vs Predicted Graph

        st.subheader("Actual vs Predicted Grades")

        comparison = results["comparison"]

        st.scatter_chart(
            comparison,
            x="Actual Grade",
            y="Predicted Grade"
        )

        st.write(
            "Compare the actual student grades "
            "with the grades predicted by the model."
        )

        st.divider()

        # Feature Importance

        st.subheader("Feature Importance")

        importance = results["importance"]

        top_features = importance.head(10)

        st.bar_chart(
            top_features.set_index("Feature")
        )

        st.write(
            "These features have the highest "
            "importance scores in the trained model."
        )

        st.divider()

        # Detailed Results

        st.subheader("Evaluation Data")

        st.dataframe(
            comparison,
            use_container_width=True
        )

        st.subheader("Feature Importance Data")

        st.dataframe(
            importance,
            use_container_width=True
        )

    except Exception as error:

        st.error(
            f"Evaluation failed: {error}"
        )
