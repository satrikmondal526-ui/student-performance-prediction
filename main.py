import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class StudentMLPipeline:
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.numeric_features = ['studytime', 'failures', 'absences', 'G1', 'G2', 'prior_grade_avg']
        self.categorical_features = ['schoolsup', 'famsup', 'internet']
        self.input_features = self.numeric_features[:5] + self.categorical_features

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f'Dataset file not found: {self.data_path}')
        df = pd.read_csv(self.data_path, sep=';')
        missing = set(self.input_features + ['G3']) - set(df.columns)
        if missing:
            raise ValueError(f'Missing columns: {missing}')
        df = df.dropna(subset=['G3'])
        if df.empty:
            raise ValueError('Dataset is empty!')
        logging.info('Dataset loaded successfully')
        return df

    def feature_engineering(self, df):
        df = df.copy()
        df['prior_grade_avg'] = (df['G1'] + df['G2']) / 2
        return df

    def preprocess_data(self, df):
        df = self.feature_engineering(df)
        X = df[self.numeric_features + self.categorical_features]
        y = df['G3']
        numeric_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('numeric', numeric_pipeline, self.numeric_features),
            ('categorical', categorical_pipeline, self.categorical_features),
        ])
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42)),
        ])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        logging.info('Data preprocessing completed')

    def train_model(self):
        if self.model is None:
            raise RuntimeError('Preprocessing must run first!')
        self.model.fit(self.X_train, self.y_train)
        logging.info('Model training completed')

    def optimize_model(self):
        if self.model is None:
            raise RuntimeError('Preprocess data first.')
        param_grid = {
            'regressor__n_estimators': [100, 150, 200, 300],
            'regressor__max_depth': [None, 5, 10, 15],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__max_features': [0.7, 1.0],
        }
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_grid,
            n_iter=15,
            cv=cv,
            scoring='neg_mean_absolute_error',
            random_state=42,
            n_jobs=-1,
            refit=True,
        )
        search.fit(self.X_train, self.y_train)
        self.model = search.best_estimator_
        print('\nMODEL OPTIMIZATION')
        print('-------------------------')
        print('Best Parameters:', search.best_params_)
        print('Cross-validation MAE:', round(-search.best_score_, 3))
        return search.best_params_

    def evaluate_model(self):
        if self.model is None:
            raise RuntimeError('Train or load the model first!')
        predictions = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, predictions)
        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        r2 = r2_score(self.y_test, predictions)
        print('\nMODEL EVALUATION')
        print('-------------------------')
        print(f'MAE: {mae:.2f}')
        print(f'RMSE: {rmse:.2f}')
        print(f'R2 Score: {r2:.2f}')
        return predictions

    def evaluate_saved_model(self):
        if self.model is None:
            raise RuntimeError('Load the model first!')
        df = self.feature_engineering(self.load_data())
        X = df[self.numeric_features + self.categorical_features]
        y = df['G3']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        predictions = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        comparison = pd.DataFrame({'Actual Grade': y_test.values, 'Predicted Grade': predictions})
        preprocessor = self.model.named_steps['preprocessor']
        regressor = self.model.named_steps['regressor']
        importance = pd.DataFrame({
            'Feature': preprocessor.get_feature_names_out(),
            'Importance': regressor.feature_importances_,
        }).sort_values(by='Importance', ascending=False)
        return {'mae': mae, 'rmse': rmse, 'r2': r2, 'comparison': comparison, 'importance': importance}

    def save_model(self):
        directory = os.path.dirname(self.model_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logging.info('Model saved successfully')

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f'Saved model not found: {self.model_path}')
        self.model = joblib.load(self.model_path)
        logging.info('Model loaded successfully')

    def predict_student(self, student_data):
        if self.model is None:
            raise RuntimeError('Train or load the model first!')
        df = pd.DataFrame([student_data])
        missing = set(self.input_features) - set(df.columns)
        if missing:
            raise ValueError(f'Missing features: {missing}')
        df['prior_grade_avg'] = (df['G1'] + df['G2']) / 2
        df = df[self.numeric_features + self.categorical_features]
        return float(self.model.predict(df)[0])

    def plot_results(self, predictions):
        plt.figure(figsize=(8, 5))
        plt.scatter(self.y_test, predictions, alpha=0.7)
        plt.xlabel('Actual Grades')
        plt.ylabel('Predicted Grades')
        plt.title('Actual vs Predicted Student Grades')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    pipeline = StudentMLPipeline(
        data_path='data/student-mat.csv',
        model_path='models/student_model.pkl',
    )
    data = pipeline.load_data()
    pipeline.preprocess_data(data)
    pipeline.optimize_model()
    predictions = pipeline.evaluate_model()
    pipeline.save_model()
    new_student = {
        'studytime': 3,
        'failures': 0,
        'absences': 2,
        'G1': 15,
        'G2': 16,
        'schoolsup': 'no',
        'famsup': 'yes',
        'internet': 'yes',
    }
    result = pipeline.predict_student(new_student)
    print('\nSTUDENT PERFORMANCE PREDICTION')
    print(f'Predicted Final Grade: {result:.2f}/20')
    pipeline.plot_results(predictions)
