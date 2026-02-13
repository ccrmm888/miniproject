from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# โหลด model
model = joblib.load("linear_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        study_hours = float(request.form["study_hours"])
        attendance_rate = float(request.form["attendance_rate"])
        late_submission_rate = float(request.form["late_submission_rate"])
        midterm_score = float(request.form["midterm_score"])

        input_data = np.array([[study_hours,
                                attendance_rate,
                                late_submission_rate,
                                midterm_score]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
