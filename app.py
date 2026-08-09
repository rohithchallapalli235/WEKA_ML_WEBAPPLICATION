import os
from flask import Flask, render_template, request, redirect, url_for, session
from weka_engine import evaluate_classifier

app = Flask(__name__)
app.secret_key = "weka_ml_app_secret_key"

DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets")

@app.route("/", methods=["GET"])
def index():
    builtin_datasets = []
    if os.path.exists(DATASET_DIR):
        builtin_datasets = [f for f in os.listdir(DATASET_DIR) if f.endswith(".arff") or f.endswith(".csv")]
        builtin_datasets.sort(key=lambda x: (0 if x == 'mushroom.arff' else 1 if 'mushroom_full' in x else 2 if 'diabetes' in x else 3 if 'breast' in x else 4))

    selected_source = session.get("last_dataset_source", "builtin")
    selected_builtin = session.get("last_builtin_dataset")

    return render_template(
        "index.html",
        builtin_datasets=builtin_datasets,
        selected_source=selected_source,
        selected_builtin=selected_builtin
    )

@app.route("/evaluate", methods=["POST"])
def evaluate():
    try:
        dataset_source = request.form.get("dataset_source", "builtin")
        algorithm = request.form.get("algorithm", "J48")
        knn_k = request.form.get("knn_k", "3")

        file_content = ""
        filename = "dataset.arff"

        if dataset_source == "builtin":
            builtin_file = request.form.get("builtin_dataset")
            if not builtin_file:
                return render_template("index.html", error="Please choose a built-in dataset.")
            filepath = os.path.join(DATASET_DIR, builtin_file)
            if not os.path.exists(filepath):
                return render_template("index.html", error=f"Builtin dataset '{builtin_file}' not found.")
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                file_content = f.read()
            filename = builtin_file
            session["last_dataset_source"] = "builtin"
            session["last_builtin_dataset"] = builtin_file

        elif dataset_source == "upload":
            if "dataset_file" not in request.files:
                return render_template("index.html", error="No file uploaded.")
            uploaded_file = request.files["dataset_file"]
            if uploaded_file.filename == "":
                return render_template("index.html", error="Selected file is empty.")
            filename = uploaded_file.filename
            file_content = uploaded_file.read().decode("utf-8", errors="ignore")
            session["last_dataset_source"] = "upload"
            session["last_builtin_dataset"] = None

        results = evaluate_classifier(
            file_content=file_content,
            filename=filename,
            algo_name=algorithm,
            params={"k": knn_k}
        )

        results["dataset_file"] = filename
        return render_template("result.html", data=results)

    except Exception as e:
        return render_template("index.html", error=f"Error evaluating dataset: {str(e)}")

@app.route("/result", methods=["GET"])
def show_result():
    results = session.get("results")
    if not results:
        return redirect(url_for("index"))
    return render_template("result.html", data=results)

if __name__ == "__main__":
    import os

    host = "0.0.0.0"
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting WEKA ML Web Application on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
