from gevent import monkey

monkey.patch_all()

from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

import io
import os
import joblib
import base64
import matplotlib.pyplot as plt
from cramer_map import cramer_mat

from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

import numpy as np
import pandas as pd
import seaborn as sns

from category_encoders import LeaveOneOutEncoder

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)

from tqdm import tqdm

pd.set_option("future.no_silent_downcasting", True)

app = Flask(__name__, template_folder="frontend")
socketio = SocketIO(app, async_mode="gevent", cors_allowed_origins="*")

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
FRONTEND_FOLDER = "frontend"
ALLOWED_EXTENSIONS = {"xlsx", "csv", "tsv"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MODEL_FOLDER"] = MODEL_FOLDER
app.config["FRONTEND_FOLDER"] = FRONTEND_FOLDER

if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_file(filepath):
    try:
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filepath.endswith(".xlsx"):
            df = pd.read_excel(filepath)
        elif filepath.endswith(".tsv"):
            df = pd.read_csv(filepath, sep="\t")
        else:
            raise ValueError("Unsupported file format")

        if df.empty:
            raise ValueError("The uploaded file is empty.")

        return df
    except pd.errors.EmptyDataError:
        raise ValueError("No columns to parse from file")
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        try:
            df = load_file(filepath)
            df_head = df.head().to_html()

            cramer_df = cramer_mat(df)

            fig, ax = plt.subplots(figsize=(9, 9))
            sns.heatmap(cramer_df, annot=True, fmt=".2f")
            plt.title("Correlation Matrix between Columns of the Dataframe")

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)

            cramer_image = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()

            json_outputs = {
                "dataframe_head": df_head,
                "filepath": filepath,
                "correlationHeatmap": cramer_image,
            }

            return jsonify(json_outputs)
        except ValueError as e:
            return jsonify({"error": str(e)})
    else:
        return jsonify({"error": "Unsupported file format"})


# Define a callable class for the callback
class ProgressCallback:
    def __init__(self, total_iterations):
        self.counter = {"iteration": 0}
        self.pbar = tqdm(total=total_iterations, desc="Training Progress")
        self.total_iterations = total_iterations

    def __call__(self, res):
        self.counter["iteration"] += 1
        self.pbar.update(1)
        progress = self.counter["iteration"] / self.total_iterations * 100
        socketio.emit("progress", {"progress": progress})

    def cleanup(self):
        self.pbar.close()
        self.pbar = None


# Function to create the ProgressCallback instance
def custom_callback(total_iterations):
    return ProgressCallback(total_iterations)


@app.route("/train", methods=["POST"])
def train_model():
    data = request.get_json()
    filepath = data["filepath"]
    task_type = data["task_type"]
    target_column = data["target_column"]
    encoding_columns = data["encoding_columns"]

    try:
        df = load_file(filepath)
    except ValueError as e:
        return jsonify({"error": str(e)})

    encoders = {}

    target_encoder = LabelEncoder()
    df[target_column] = target_encoder.fit_transform(df[target_column])
    encoders[target_column] = target_encoder

    for column, enc_type in encoding_columns.items():
        if enc_type == "ordinal":
            encoder = OrdinalEncoder()
            df[column] = encoder.fit_transform(df[column])
        elif enc_type == "loo":
            encoder = LeaveOneOutEncoder(cols=[column], sigma=0.1)
            df[column] = encoder.fit_transform(df[column], df[target_column])
        else:
            encoder = LabelEncoder()
            df[column] = encoder.fit_transform(df[column])

        encoders[column] = encoder

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    param_dist = {
        "n_estimators": Integer(10, 200),
        "max_depth": Integer(5, 20),
        "min_samples_split": Integer(2, 11),
        "min_samples_leaf": Integer(1, 11),
        "max_features": Categorical(["sqrt", "log2", None]),
    }

    if task_type == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y
        )
        cv_ = StratifiedKFold(n_splits=5, shuffle=True)
        scoring_ = "f1"
        model = RandomForestClassifier(
            bootstrap=True, class_weight="balanced", n_jobs=4
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        cv_ = KFold(n_splits=5, shuffle=True)
        scoring_ = "neg_mean_squared_error"
        model = RandomForestRegressor(bootstrap=True, n_jobs=4)

    bayes_search = BayesSearchCV(
        estimator=model,
        search_spaces=param_dist,
        n_iter=50,
        scoring=scoring_,
        cv=cv_,
        n_jobs=-1,
    )

    callback_fn = custom_callback(total_iterations=50)
    bayes_search.fit(X_train, y_train, callback=callback_fn)

    # Cleanup the callback
    callback_fn.cleanup()

    # Manually remove the callback and tqdm object references
    del callback_fn

    y_eval = bayes_search.predict(X_test)

    if task_type == "classification":
        accuracy = accuracy_score(y_test, y_eval)
        precision = precision_score(y_test, y_eval, average="weighted")
        recall = recall_score(y_test, y_eval, average="weighted")

        accuracy_formatted = "{:.2f}%".format(accuracy * 100)
        precision_formatted = "{:.2f}%".format(precision * 100)
        recall_formatted = "{:.2f}%".format(recall * 100)
        metrics = pd.DataFrame.from_dict(
            {
                "metrics": ["accuracy", "precision", "recall"],
                "values": [accuracy_formatted, precision_formatted, recall_formatted],
            }
        )
    else:
        mae = mean_absolute_error(y_test, y_eval)
        mape = mean_absolute_percentage_error(y_test, y_eval)
        mse = mean_squared_error(y_test, y_eval)
        rmse = mean_squared_error(y_test, y_eval, squared=False)

        mae_formatted = "{:.2f}".format(mae)
        mape_formatted = "{:.2f}%".format(mape)
        mse_formatted = "{:.2f}".format(mse)
        rmse_formatted = "{:.2f}".format(rmse)
        metrics = pd.DataFrame.from_dict(
            {
                "metrics": ["mae", "mape", "mse", "rmse"],
                "values": [
                    mae_formatted,
                    mape_formatted,
                    mse_formatted,
                    rmse_formatted,
                ],
            }
        )

    df_metrics = metrics.to_html()

    # Feature Importance Bar Plot
    feature_importances = bayes_search.best_estimator_.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": feature_importances}
    ).sort_values(by="importance", ascending=False)

    # Plotting with seaborn and using io.BytesIO to avoid file I/O
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.barplot(x="importance", y="feature", data=importance_df, ax=ax)
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    # Convert image to base64
    importance_image = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    model_path = os.path.join(app.config["MODEL_FOLDER"], "encoder_and_trained_rf.pkl")

    with open(model_path, "wb") as f:
        joblib.dump((bayes_search, encoders), f, compress="lz4")

    socketio.emit(
        "train_complete", {"importance": importance_image, "metrics": df_metrics}
    )

    json_outputs = {
        "message": "Model trained successfully",
        "model_path": model_path,
    }

    return jsonify(json_outputs)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    model_path = request.form["model_path"]
    target_column = request.form["target_column"]
    task_type = request.form["task_type"]

    if file.filename == "":
        return jsonify({"error": "No selected file"})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        try:
            df = load_file(filepath)
        except ValueError as e:
            return jsonify({"error": str(e)})

        # Load the trained model and encoders
        model, encoders = joblib.load(model_path)

        # Encode the input dataframe with the same encoders used during training
        for column, encoder in encoders.items():
            if column in df.columns:
                df[column] = encoder.transform(df[column])

        # Make predictions
        prediction = model.predict(df)

        # Decode the predictions if the target column is categorical
        if task_type == "classification" and target_column in encoders:
            prediction = encoders[target_column].inverse_transform(prediction)

        # Since the input is a single row, return the single prediction value
        prediction_value = (
            prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction
        )
        displayed_prediction = target_column + ": " + str(prediction_value)

        return jsonify({"prediction": displayed_prediction})

    return jsonify({"error": "Unsupported file format"})


@app.route("/frontend/<path:filename>")
def static_files(filename):
    return send_from_directory(app.config["FRONTEND_FOLDER"], filename)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])

    if not os.path.exists(app.config["MODEL_FOLDER"]):
        os.makedirs(app.config["MODEL_FOLDER"])

    server = pywsgi.WSGIServer(("0.0.0.0", 8000), app, handler_class=WebSocketHandler)
    socketio.run(app, host="0.0.0.0", port=8000, debug=True)
