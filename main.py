
import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# 1. Logging Configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)


# 2. Create ML Pipeline Class

class StudentMLPipeline:

    def __init__(self, data_path, model_path):

        self.data_path = data_path
        self.model_path = model_path
        self.model = None

        self.numeric_features = [
            "studytime",
            "failures",
            "absences",
            "G1",
            "G2",
            "prior_grade_avg"
        ]

        self.categorical_features = [
            "schoolsup",
            "famsup",
            "internet"
        ]

        self.input_features = (
            self.numeric_features[:5] +
            self.categorical_features
        )

    # 3. Data Ingestion

    def load_data(self):

        try:

            if not os.path.exists(self.data_path):
                raise FileNotFoundError(
                    "Dataset file not found!"
                )

            df = pd.read_csv(
                self.data_path,
                sep=";"
            )

            required_columns = (
                self.input_features + ["G3"]
            )

            missing = (
                set(required_columns) -
                set(df.columns)
            )

            if missing:
                raise ValueError(
                    f"Missing columns: {missing}"
                )

            df = df.dropna(subset=["G3"])

            if df.empty:
                raise ValueError("Dataset is empty!")

            logging.info(
                "Dataset loaded successfully"
            )

            print("\nDataset Preview:")
            print(df.head())

            print("\nDataset Shape:", df.shape)

            return df

        except Exception as error:

            logging.error(
                f"Data loading failed: {error}"
            )

            raise

    # 4. Feature Engineering

    def feature_engineering(self, df):

        df = df.copy()

        df["prior_grade_avg"] = (
            df["G1"] + df["G2"]
        ) / 2

        logging.info(
            "Feature engineering completed"
        )

        return df

    # 5. Data Preprocessing

    def preprocess_data(self, df):

        df = self.feature_engineering(df)

        X = df[
            self.numeric_features +
            self.categorical_features
        ]

        y = df["G3"]

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                )
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    self.numeric_features
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    self.categorical_features
                )
            ]
        )

        self.model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    RandomForestRegressor(
                        n_estimators=100,
                        random_state=42
                    )
                )
            ]
        )

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        ) = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        logging.info(
            "Data preprocessing completed"
        )

    # 6. Model Training

    def train_model(self):

        if self.model is None:
            raise RuntimeError(
                "Preprocessing must run first!"
            )

        self.model.fit(
            self.X_train,
            self.y_train
        )

        logging.info(
            "Model training completed"
        )

    # 7. Model Evaluation

    def evaluate_model(self):

        predictions = self.model.predict(
            self.X_test
        )

        mae = mean_absolute_error(
            self.y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                self.y_test,
                predictions
            )
        )

        r2 = r2_score(
            self.y_test,
            predictions
        )

        print("\nMODEL EVALUATION")
        print("-------------------------")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.2f}")

        return predictions

    # 12. Evaluate Saved Model

    def evaluate_saved_model(self):

        df = self.load_data()
        df = self.feature_engineering(df)

        X = df[
            self.numeric_features +
            self.categorical_features
        ]

        y = df["G3"]

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42
            )
        )

        predictions = self.model.predict(X_test)

        mae = mean_absolute_error(
            y_test, predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test, predictions
            )
        )

        r2 = r2_score(
            y_test, predictions
        )

        actual_vs_predicted = pd.DataFrame({
            "Actual Grade": y_test.values,
            "Predicted Grade": predictions
        })

        preprocessor = self.model.named_steps[
            "preprocessor"
        ]

        regressor = self.model.named_steps[
            "regressor"
        ]

        feature_names = (
            preprocessor.get_feature_names_out()
        )

        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": regressor.feature_importances_
        }).sort_values(
            by="Importance",
            ascending=False
        )

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "comparison": actual_vs_predicted,
            "importance": importance
        }


    # 8. Save Model

    def save_model(self):

        os.makedirs(
            os.path.dirname(self.model_path),
            exist_ok=True
        )

        joblib.dump(
            self.model,
            self.model_path
        )

        logging.info(
            "Model saved successfully"
        )

    # 9. Load Model

    def load_model(self):

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "Saved model not found!"
            )

        self.model = joblib.load(
            self.model_path
        )

        logging.info(
            "Model loaded successfully"
        )

    # 10. Student Prediction

    def predict_student(self, student_data):

        if self.model is None:
            raise RuntimeError(
                "Train or load the model first!"
            )

        df = pd.DataFrame([student_data])

        missing = (
            set(self.input_features) -
            set(df.columns)
        )

        if missing:
            raise ValueError(
                f"Missing features: {missing}"
            )

        df["prior_grade_avg"] = (
            df["G1"] + df["G2"]
        ) / 2

        df = df[
            self.numeric_features +
            self.categorical_features
        ]

        prediction = self.model.predict(df)

        return float(prediction[0])

    # 11. Visualization

    def plot_results(self, predictions):

        plt.figure(figsize=(8, 5))

        plt.scatter(
            self.y_test,
            predictions,
            alpha=0.7
        )

        plt.xlabel("Actual Grades")
        plt.ylabel("Predicted Grades")

        plt.title(
            "Actual vs Predicted Student Grades"
        )

        plt.grid(True)
        plt.tight_layout()
        plt.show()


# 12. Main Execution

if __name__ == "__main__":

    pipeline = StudentMLPipeline(
        data_path="data/student-mat.csv",
        model_path="models/student_model.pkl"
    )

    # Load Dataset
    data = pipeline.load_data()

    # Preprocess Data
    pipeline.preprocess_data(data)

    # Train Model
    pipeline.train_model()

    # Evaluate Model
    predictions = pipeline.evaluate_model()

    # Save Model
    pipeline.save_model()

    # New Student Information
    new_student = {

        "studytime": 3,
        "failures": 0,
        "absences": 2,
        "G1": 15,
        "G2": 16,
        "schoolsup": "no",
        "famsup": "yes",
        "internet": "yes"

    }

    # Predict Grade
    result = pipeline.predict_student(
        new_student
    )

    print("\nSTUDENT PERFORMANCE PREDICTION")

    print(
        f"Predicted Final Grade: {result:.2f}/20"
    )

    # Display Graph
    pipeline.plot_results(predictions)
