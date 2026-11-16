
from main import StudentMLPipeline

pipeline = StudentMLPipeline(
    "data/student-mat.csv",
    "models/student_model.pkl"
)

pipeline.load_model()

student = {
    "studytime": 2,
    "failures": 0,
    "absences": 5,
    "G1": 12,
    "G2": 14,
    "schoolsup": "no",
    "famsup": "yes",
    "internet": "yes"
}

result = pipeline.predict_student(student)

print("Predicted Grade:", round(result, 2))
