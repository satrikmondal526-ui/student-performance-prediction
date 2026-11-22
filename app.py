import os
from pathlib import Path

import pandas as pd
import streamlit as st
from main import StudentMLPipeline

try:
    import psycopg2
except ImportError:
    psycopg2 = None

BASE = Path(__file__).resolve().parent
DATA_PATH = str(BASE / 'data' / 'student-mat.csv')
MODEL_PATH = str(BASE / 'models' / 'student_model.pkl')

st.set_page_config(page_title='Student Performance AI', page_icon='🎓', layout='wide')
st.markdown('''<style>
.block-container {padding-top: 2rem;}
.dashboard-title {font-size: 34px; font-weight: 800; color: #243b53;}
.dashboard-subtitle {color: #627d98; font-size: 16px;}
.result-card {background: #e8f5e9; padding: 24px; border-radius: 14px; text-align: center; color: #14532d;}
.result-number {font-size: 42px; font-weight: 800;}
</style>''', unsafe_allow_html=True)

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
        raise RuntimeError('Install psycopg2-binary to use prediction history.')
    try:
        url = st.secrets['postgres']['url']
    except (KeyError, FileNotFoundError) as exc:
        raise RuntimeError('Missing PostgreSQL URL in .streamlit/secrets.toml or Streamlit Cloud secrets.') from exc
    return psycopg2.connect(url, sslmode='require', connect_timeout=10)


def save_prediction(record):
    sql = '''INSERT INTO predictions
        (studytime, failures, absences, g1, g2, schoolsup, famsup, internet, predicted_grade)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)'''
    values = (
        record['studytime'], record['failures'], record['absences'],
        record['G1'], record['G2'], record['schoolsup'],
        record['famsup'], record['internet'], record['predicted_grade'],
    )
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, values)


def get_prediction_history():
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute('SELECT * FROM predictions ORDER BY prediction_date DESC')
            rows = cursor.fetchall()
            columns = [column[0] for column in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def clear_prediction_history():
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute('DELETE FROM predictions')


st.sidebar.title('🎓 Student AI')
st.sidebar.write('Machine Learning Dashboard')
page = st.sidebar.radio('Navigation', ['Dashboard', 'Model Evaluation', 'About Project'])
st.sidebar.divider()
st.sidebar.success('Model: Random Forest')

if page == 'Dashboard':
    st.markdown('<div class="dashboard-title">Student Performance Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="dashboard-subtitle">AI-powered student academic performance estimation</div>', unsafe_allow_html=True)
    st.divider()

    try:
        pipeline = load_model()
    except Exception as error:
        st.error(f'Model loading failed: {error}')
        st.stop()

    st.subheader('📋 Student Information')
    with st.form('prediction_form'):
        left, right = st.columns(2)
        with left:
            studytime = st.selectbox('Weekly Study Time', [1, 2, 3, 4], format_func=lambda x: {
                1: 'Less than 2 hours', 2: '2–5 hours',
                3: '5–10 hours', 4: 'More than 10 hours'
            }[x])
            failures = st.number_input('Previous Failures', min_value=0, max_value=4, value=0)
            absences = st.number_input('Number of Absences', min_value=0, max_value=100, value=2)
            schoolsup = st.selectbox('Extra School Support', ['no', 'yes'])
        with right:
            g1 = st.slider('First Period Grade (G1)', 0, 20, 15)
            g2 = st.slider('Second Period Grade (G2)', 0, 20, 16)
            famsup = st.selectbox('Family Support', ['yes', 'no'])
            internet = st.selectbox('Internet Access', ['yes', 'no'])
        submitted = st.form_submit_button('🚀 Predict Student Performance', type='primary', use_container_width=True)

    if submitted:
        student_data = {
            'studytime': studytime, 'failures': failures, 'absences': absences,
            'G1': g1, 'G2': g2, 'schoolsup': schoolsup,
            'famsup': famsup, 'internet': internet,
        }
        try:
            prediction = pipeline.predict_student(student_data)
            st.session_state['last_prediction'] = {
                'prediction': prediction, 'G1': g1, 'G2': g2,
            }
            try:
                save_prediction({**student_data, 'predicted_grade': round(prediction, 2)})
                st.success('Prediction saved to PostgreSQL.')
            except Exception as error:
                st.warning(f'Prediction calculated, but cloud history was not saved: {error}')
        except Exception as error:
            st.error(f'Prediction failed: {error}')

    if 'last_prediction' in st.session_state:
        result = st.session_state['last_prediction']
        prediction = result['prediction']
        st.subheader('📊 Prediction Results')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="result-card"><h3>Predicted Final Grade</h3><div class="result-number">{prediction:.2f}/20</div><p>Estimated academic score</p></div>', unsafe_allow_html=True)
        with col2:
            st.metric('Predicted Grade', f'{prediction:.2f}/20')
            st.progress(min(max(prediction / 20, 0.0), 1.0))
            st.write(f'Estimated score: {prediction / 20 * 100:.1f}%')
        chart_data = pd.DataFrame({
            'Category': ['First Period', 'Second Period', 'Predicted Final'],
            'Grade': [result['G1'], result['G2'], prediction],
        }).set_index('Category')
        st.subheader('📈 Grade Visualization')
        st.bar_chart(chart_data)
        st.info('This is an educational estimate, not a definitive assessment of student ability.')

    st.divider()
    st.subheader('🗄️ Cloud Prediction History')
    try:
        history = get_prediction_history()
    except Exception as error:
        st.warning(f'Prediction history unavailable: {error}')
    else:
        if history.empty:
            st.info('No prediction records found yet.')
        else:
            st.metric('Total Predictions', len(history))
            st.dataframe(history, use_container_width=True)
            st.download_button('📥 Download CSV Report', history.to_csv(index=False).encode('utf-8'),
                               file_name='student_prediction_history.csv', mime='text/csv')
            with st.expander('Database maintenance'):
                confirm = st.checkbox('I understand this will delete all stored prediction records.')
                if st.button('🗑️ Clear All Prediction Records', disabled=not confirm):
                    try:
                        clear_prediction_history()
                        st.success('Prediction records cleared.')
                        st.rerun()
                    except Exception as error:
                        st.error(f'Could not clear records: {error}')

elif page == 'Model Evaluation':
    st.title('📊 Model Evaluation Dashboard')
    st.write('Evaluate the saved Random Forest Regression model.')
    try:
        pipeline = load_model()
        results = pipeline.evaluate_saved_model()
    except Exception as error:
        st.error(f'Evaluation failed: {error}')
    else:
        a, b, c = st.columns(3)
        a.metric('MAE', f"{results['mae']:.2f}")
        b.metric('RMSE', f"{results['rmse']:.2f}")
        c.metric('R² Score', f"{results['r2']:.3f}")
        st.caption('Lower MAE and RMSE indicate smaller errors; R² describes explained variance.')
        st.divider()
        st.subheader('Actual vs Predicted Grades')
        st.scatter_chart(results['comparison'], x='Actual Grade', y='Predicted Grade')
        st.subheader('Top 10 Feature Importances')
        importance = results['importance'].head(10)
        st.bar_chart(importance.set_index('Feature')['Importance'])
        st.subheader('Evaluation Data')
        st.dataframe(results['comparison'], use_container_width=True)
        st.subheader('Feature Importance Data')
        st.dataframe(results['importance'], use_container_width=True)

elif page == 'About Project':
    st.title('ℹ️ About This Project')
    st.write('Student Performance Prediction is an end-to-end machine learning project that estimates a final grade from academic and support-related inputs.')
    st.subheader('Technologies Used')
    st.write('Python · Pandas · NumPy · Scikit-learn · Random Forest Regression · Streamlit · Joblib · PostgreSQL')
    st.divider()
    st.subheader('Machine Learning Workflow')
    for index, step in enumerate([
        'Data Collection', 'Data Preprocessing', 'Feature Engineering',
        'Model Training', 'Model Evaluation', 'Model Saving', 'Grade Prediction',
    ], start=1):
        st.write(f'{index}. {step}')
    st.caption('Student Performance Prediction | End-to-End ML Pipeline Project')
