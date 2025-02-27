from flask import Flask, render_template, request, redirect, url_for, session, send_file
import pandas as pd
import pickle
from xgboost import XGBClassifier

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session management

# Load the trained XGBoost model
with open("xgb_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/", methods=["GET"])
def index():
    # Redirect to the login page initially
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # Validate credentials (for simplicity, using "admin" as the user)
        username = request.form.get("username")
        password = request.form.get("password")
        
        if username == "admin" and password == "password":  # Simple check
            session["user"] = username  # Set session if valid login
            return redirect(url_for("dashboard"))
        else:
            return "Invalid credentials. Please try again."

    return render_template("login.html")

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))  # Ensure user is logged in

    if request.method == "POST":
        # Handle file upload
        uploaded_file = request.files["file"]
        if uploaded_file:
            if uploaded_file.filename.endswith('.csv'):  # Ensure it's a CSV file
                data = pd.read_csv(uploaded_file)

                # Preprocess the data
                if "Amount" in data.columns:
                    data["Amount"] = (data["Amount"] - data["Amount"].mean()) / data["Amount"].std()

                # Prediction
                X = data.drop(columns=["Class"], errors="ignore")  # Ensure 'Class' is dropped if present
                predictions = model.predict(X)
                data["Prediction"] = predictions
                data["Risk Score"] = model.predict_proba(X)[:, 1]

                # Flagged transactions
                flagged_data = data[data["Prediction"] == 1]

                # Save for download
                flagged_data.to_csv("flagged_transactions.csv", index=False)

                return render_template(
                    "dashboard.html",
                    total=len(data),
                    flagged=len(flagged_data),
                    flagged_data=flagged_data.to_dict(orient="records"),
                )
            else:
                return "Invalid file format. Please upload a CSV file.", 400

    return render_template("dashboard.html")

@app.route("/download", methods=["GET"])
def download():
    if "user" not in session:
        return redirect(url_for("login"))

    return send_file("flagged_transactions.csv", as_attachment=True)

@app.route("/logout")
def logout():
    session.pop("user", None)  # Remove the user from the session
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
