from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__, template_folder="templates")

# Load the trained SVC model and preprocessors
with open(r"C:\Users\sathv\OneDrive\Desktop\capstone_project\Execution\newfolder\svc_model.pkl", "rb") as f:
    model, scaler, label_encoders = pickle.load(f)

@app.route("/")
def home():
    return render_template("survey.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check if request is JSON (from JS) or form (from regular form submission)
        if request.is_json:
            form_data = request.get_json()
        else:
            form_data = request.form.to_dict()

        # Convert categorical values using Label Encoders
        for col in ["gender", "ethnicity", "jaundice", "austim", "used_app_before", "relation"]:
            if col in form_data:
                try:
                    form_data[col] = label_encoders[col].transform([form_data[col]])[0]
                except ValueError:  # Handle unseen categories
                    form_data[col] = 0

        # Convert Yes/No answers to 1/0
        for key in form_data:
            if form_data[key] == "Yes":
                form_data[key] = 1
            elif form_data[key] == "No":
                form_data[key] = 0
            else:
                try:
                    form_data[key] = int(form_data[key])  # Convert numeric values
                except ValueError:
                    form_data[key] = 0  # Default fallback

        # Convert to NumPy array and scale
        user_data = np.array(list(form_data.values())).reshape(1, -1)
        user_data = scaler.transform(user_data)

        # Make the prediction
        prediction = model.predict(user_data)[0]

        # Check if the model supports probability predictions
        if hasattr(model, "predict_proba"):
            prediction_prob = model.predict_proba(user_data)[0][1]
        else:
            prediction_prob = 0.5  # Default probability

        # Generate a readable result
        result_text = (
            f"High likelihood of Autism (Confidence: {prediction_prob:.2f})"
            if prediction == 1
            else f"Low likelihood of Autism (Confidence: {1 - prediction_prob:.2f})"
        )

        # Return JSON response if AJAX request, otherwise render the result page
        if request.is_json:
            return jsonify({"result": result_text})
        else:
            return render_template("result.html", result=result_text)

    except Exception as e:
        return f"Error processing prediction: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
