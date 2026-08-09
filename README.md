# WEKA Machine Learning Web Application

A full-stack Web Application built for VS Code that executes Machine Learning algorithms (**J48, Naive Bayes, KNN, ID3**) on ARFF and CSV datasets, rendering vibrant **Colorful Decision Trees**, **Confusion Matrices**, **Classification Reports**, and **Multi-Class ROC Curves**.

---

## 🌟 Key Features

1. **Four Core Algorithms**:
   - **J48 (C4.5)**: Pruned Decision Tree with Information Gain & Entropy.
   - **Naive Bayes**: Probabilistic classifier.
   - **KNN (IBk)**: K-Nearest Neighbors Classifier with configurable $K$.
   - **ID3**: Information Gain Unpruned Decision Tree.
2. **Inbuilt Datasets**:
   - `mushroom.arff`: Pre-loaded Mushroom classification dataset.
   - `employee_salary.arff`: Pre-loaded Growth & Seed Variety dataset matching Reference Image #1.
3. **File Manager Dataset Loader**:
   - Easily upload any `.arff` or `.csv` dataset directly from your file manager.
4. **Rich Visual Output (Matching Your Reference Screenshots)**:
   - **Green Success Banner**: `Algorithm Executed Successfully` & Accuracy.
   - **Confusion Matrix**: Cell heatmaps showing class predictions.
   - **Classification Report**: Precision, Recall, F1-Score, Support per class.
   - **Multi-Class ROC Curves**: Interactive curve graph with legend AUC scores.
   - **Colorful Decision Tree Visualizer** (for J48 & ID3):
     - Colorful gradient oval decision nodes.
     - Rectangular leaf nodes with `Class (Count)` format.
     - Branch condition labels (`= Poor`, `= Average`, `= Good`, etc.).
     - Bottom Node Information status panel displaying **Number of Leaves** and **Size of the Tree**.
5. **WEKA CLI Integration**:
   - Runs out-of-the-box using built-in Python ML engine.
   - Connects seamlessly to your local WEKA installation (`weka.jar` CLI) if Java and WEKA are installed on your machine.

---

## 🚀 Quick Start Guide for VS Code

### Step 1: Open Folder in VS Code
1. Open **Visual Studio Code**.
2. Click **File > Open Folder...**
3. Select the project folder:
   `C:\Users\rohit\.gemini\antigravity\scratch\weka_ml_webapp`

### Step 2: Set Active Workspace
Set `C:\Users\rohit\.gemini\antigravity\scratch\weka_ml_webapp` as your VS Code workspace directory.

### Step 3: Run the Application
You can run the web app in two ways:
- **Option A (VS Code F5 Key)**: Press `F5` in VS Code. The pre-configured debug launcher `.vscode/launch.json` will start the server.
- **Option B (Terminal Command)**: Open the VS Code Terminal (`Ctrl + ~`) and run:
  ```bash
  python app.py
  ```

### Step 4: Open Web App
Open your web browser and navigate to:
`http://127.0.0.1:5000`

---

## 🌍 Deploying through GitHub
GitHub itself stores your project, but dynamic Flask apps need a Python host. The simplest flow is:

1. Create a GitHub repository and push this project.
2. Connect the repository to a cloud host such as **Render**, **Railway**, or **Fly.io**.
3. Use these settings:
   - Build command: `pip install -r requirements.txt`
   - Start command: `python app.py`
4. The app now runs on a public URL provided by the host.

> Note: GitHub Pages is only for static sites, so use a Python host for this Flask app.

---

## 🔗 Connecting to Local WEKA (Optional)

If you have **WEKA** installed on your PC:
1. Ensure Java is installed and added to system `PATH`.
2. Place `weka.jar` in `C:\Program Files\Weka-3-8-6\weka.jar` or in the project folder.
3. The web app automatically detects local WEKA and updates the status indicator in the top navbar!
