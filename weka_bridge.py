import subprocess
import os
import shutil

def check_java_installed():
    """Checks if Java CLI is installed and accessible in current PATH."""
    java_path = shutil.which("java")
    if not java_path:
        return False, "Java environment variable not found in PATH."
    try:
        res = subprocess.run(["java", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return True, res.stderr.split('\n')[0]
    except Exception as e:
        return False, str(e)

def find_weka_jar():
    """Scans common directories or WEKA_HOME for weka.jar."""
    common_paths = [
        os.environ.get("WEKA_HOME", ""),
        "C:\\Program Files\\Weka-3-8-6\\weka.jar",
        "C:\\Program Files\\Weka-3-9-6\\weka.jar",
        "C:\\Program Files\\Weka-3-8\\weka.jar",
        "C:\\weka\\weka.jar",
        "./weka.jar"
    ]
    for path in common_paths:
        if path and os.path.isfile(path):
            return path
    return None

def execute_weka_cli(arff_filepath, algorithm="J48"):
    """
    Executes native WEKA jar CLI command if weka.jar and java are found.
    Map algorithm name to WEKA classifier class.
    """
    java_ok, java_info = check_java_installed()
    weka_jar = find_weka_jar()

    if not java_ok or not weka_jar:
        return {
            'available': False,
            'message': f"Local WEKA CLI unavailable ({java_info if not java_ok else 'weka.jar not found'}). Using built-in Python WEKA engine."
        }

    weka_class_map = {
        'J48': 'weka.classifiers.trees.J48',
        'ID3': 'weka.classifiers.trees.Id3',
        'Naive Bayes': 'weka.classifiers.bayes.NaiveBayes',
        'KNN': 'weka.classifiers.lazy.IBk'
    }

    classifier_cls = weka_class_map.get(algorithm, 'weka.classifiers.trees.J48')

    cmd = [
        "java", "-cp", weka_jar, classifier_cls,
        "-t", arff_filepath
    ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        return {
            'available': True,
            'output': res.stdout,
            'error': res.stderr
        }
    except Exception as e:
        return {
            'available': False,
            'message': f"Error running WEKA command: {str(e)}"
        }
