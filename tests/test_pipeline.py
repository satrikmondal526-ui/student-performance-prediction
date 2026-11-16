
import pandas as pd

from main import StudentMLPipeline


def create_sample_data():

    return pd.DataFrame({

        "studytime": [
            1, 2, 3, 4, 2,
            3, 1, 4, 2, 3,
            1, 2, 3, 4, 2,
            3, 1, 4, 2, 3
        ],

        "failures": [
            0, 0, 1, 0, 0,
            0, 2, 0, 1, 0,
            0, 1, 0, 0, 1,
            0, 2, 0, 0, 1
        ],

        "absences": [
            2, 4, 0, 1, 3,
            5, 8, 1, 2, 0,
            4, 3, 1, 2, 5,
            0, 6, 2, 1, 3
        ],

        "G1": [
            10, 12, 15, 18, 14,
            16, 8, 19, 13, 17,
            11, 14, 16, 18, 12,
            15, 9, 20, 13, 17
        ],

        "G2": [
            11, 13, 16, 17, 15,
            17, 9, 18, 14, 16,
            12, 15, 17, 19, 13,
            16, 10, 19, 14, 18
        ],

        "schoolsup": [
            "yes", "no", "no", "yes", "no",
            "yes", "no", "no", "yes", "no",
            "no", "yes", "no", "yes", "no",
            "yes", "no", "no", "yes", "no"
        ],

        "famsup": [
            "yes", "yes", "no", "yes", "no",
            "yes", "no", "yes", "yes", "no",
            "yes", "no", "yes", "yes", "no",
            "yes", "no", "yes", "no", "yes"
        ],

        "internet": [
            "yes", "yes", "yes", "yes", "no",
            "yes", "no", "yes", "yes", "yes",
            "no", "yes", "yes", "yes", "no",
            "yes", "no", "yes", "yes", "yes"
        ],

        "G3": [
            11, 13, 16, 18, 15,
            17, 9, 19, 14, 17,
            12, 15, 17, 19, 13,
            16, 10, 19, 14, 18
        ]

    })


def test_feature_engineering(tmp_path):

    pipeline = StudentMLPipeline(
        "data/student-mat.csv",
        str(tmp_path / "model.pkl")
    )

    df = create_sample_data()

    result = pipeline.feature_engineering(df)

    assert "prior_grade_avg" in result.columns

    assert result.loc[0, "prior_grade_avg"] == 10.5


def test_model_training_and_prediction(tmp_path):

    pipeline = StudentMLPipeline(
        "data/student-mat.csv",
        str(tmp_path / "model.pkl")
    )

    df = create_sample_data()

    pipeline.preprocess_data(df)

    pipeline.train_model()

    student = {

        "studytime": 3,
        "failures": 0,
        "absences": 2,
        "G1": 15,
        "G2": 16,
        "schoolsup": "no",
        "famsup": "yes",
        "internet": "yes"

    }

    prediction = pipeline.predict_student(student)

    assert isinstance(prediction, float)

    assert 0 <= prediction <= 20


def test_model_save_and_load(tmp_path):

    model_path = str(tmp_path / "student_model.pkl")

    pipeline = StudentMLPipeline(
        "data/student-mat.csv",
        model_path
    )

    df = create_sample_data()

    pipeline.preprocess_data(df)

    pipeline.train_model()

    pipeline.save_model()

    assert pipeline.model is not None

    # Load saved model into a new object

    new_pipeline = StudentMLPipeline(
        "data/student-mat.csv",
        model_path
    )

    new_pipeline.load_model()

    assert new_pipeline.model is not None
