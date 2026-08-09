import os
from weka_engine import evaluate_classifier

def run_comprehensive_tests():
    datasets = [
        ('mushroom_full.arff', 'Mushroom Full (8,124 instances)'),
        ('diabetes.arff', 'Pima Diabetes (768 instances)'),
        ('breast-cancer.arff', 'UCI Breast Cancer (286 instances)'),
        ('employee_salary.arff', 'Employee Salary (Reference Tree)')
    ]

    for ds_file, ds_label in datasets:
        filepath = os.path.join('datasets', ds_file)
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'r') as f:
            content = f.read()

        print(f"\n=======================================================")
        print(f" Dataset: {ds_label}")
        print(f"=======================================================")

        for algo in ['J48', 'ID3', 'Naive Bayes', 'KNN']:
            res = evaluate_classifier(content, ds_file, algo, params={'eval_mode': 'cross_val'})
            acc = res['accuracy']
            leaves = res['tree_data']['num_leaves'] if res['tree_data'] else 'N/A'
            tree_size = res['tree_data']['tree_size'] if res['tree_data'] else 'N/A'
            roc_aucs = [f"{r['class']}: {r['auc']}" for r in res['roc_curves']] if res['roc_curves'] else []

            print(f"Algo: {algo:12} | Acc: {acc:6.2f}% | Leaves: {leaves:>3} | TreeSize: {tree_size:>3} | ROC AUCs: {', '.join(roc_aucs[:2])}")

if __name__ == "__main__":
    run_comprehensive_tests()
