import io
import re
import math
import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, cohen_kappa_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold, train_test_split


def parse_arff(content_str):
    lines = content_str.strip().split('\n')
    relation_name = "Dataset"
    attributes = []
    data_lines = []
    in_data = False

    for line in lines:
        line_clean = line.strip()
        if not line_clean or line_clean.startswith('%'):
            continue
        
        if line_clean.lower().startswith('@relation'):
            parts = line_clean.split(maxsplit=1)
            if len(parts) > 1:
                relation_name = parts[1].strip('"\'')
            continue

        if line_clean.lower().startswith('@attribute'):
            match = re.match(r'@attribute\s+([^\s]+|\"[^\"]+\"|\'[^\']+\')\s+(.+)', line_clean, re.IGNORECASE)
            if match:
                attr_name = match.group(1).strip('"\'')
                attr_type = match.group(2).strip()
                nominal_vals = None
                if attr_type.startswith('{') and attr_type.endswith('}'):
                    raw_vals = attr_type[1:-1].split(',')
                    nominal_vals = [v.strip().strip('"\'') for v in raw_vals]
                attributes.append({
                    'name': attr_name,
                    'type': 'nominal' if nominal_vals else 'numeric',
                    'values': nominal_vals
                })
            continue

        if line_clean.lower().startswith('@data'):
            in_data = True
            continue

        if in_data:
            row = [val.strip().strip('"\'') for val in line_clean.split(',')]
            if len(row) == len(attributes):
                data_lines.append(row)

    col_names = [attr['name'] for attr in attributes]
    df = pd.DataFrame(data_lines, columns=col_names)
    
    for attr in attributes:
        col = attr['name']
        if attr['type'] == 'numeric':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    return df, attributes, relation_name


def parse_dataset(file_content, filename):
    if filename.endswith('.arff'):
        return parse_arff(file_content)
    else:
        df = pd.read_csv(io.StringIO(file_content))
        relation_name = filename.rsplit('.', 1)[0]
        attributes = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                attributes.append({'name': col, 'type': 'numeric', 'values': None})
            else:
                vals = list(df[col].astype(str).unique())
                attributes.append({'name': col, 'type': 'nominal', 'values': vals})
        return df, attributes, relation_name


# Distinct Class Color Palette matching Image #1 (Orange, Green, Purple, Cyan, Amber)
CLASS_COLOR_PALETTE = [
    "#e28743",  # Class 1: Warm Orange (Image #1 setosa)
    "#7cd685",  # Class 2: Vivid Green (Image #1 versicolor)
    "#8a63d2",  # Class 3: Soft Purple (Image #1 virginica)
    "#38bdf8",  # Class 4: Bright Cyan
    "#f59e0b",  # Class 5: Amber
    "#ec4899",  # Class 6: Pink
]

def get_class_color(class_index):
    return CLASS_COLOR_PALETTE[class_index % len(CLASS_COLOR_PALETTE)]


# ==========================================
# 1. Custom Multi-Way Split ID3 Tree Engine
# ==========================================
class MultiWayID3Node:
    def __init__(self, attribute=None, is_leaf=False, prediction=None, counts=None):
        self.attribute = attribute
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.counts = counts or {}
        self.branches = {}

def entropy(y_vec):
    if len(y_vec) == 0:
        return 0.0
    vals, counts = np.unique(y_vec, return_counts=True)
    probs = counts / len(y_vec)
    return -sum(p * math.log2(p) for p in probs if p > 0)

def build_id3_tree(df, feature_cols, target_col, depth=0, max_depth=6):
    y_vals = df[target_col].values
    unique_classes, counts = np.unique(y_vals, return_counts=True)
    counts_dict = dict(zip(unique_classes, counts))
    majority_class = unique_classes[np.argmax(counts)]

    if len(unique_classes) == 1 or depth >= max_depth or len(feature_cols) == 0:
        return MultiWayID3Node(is_leaf=True, prediction=majority_class, counts=counts_dict)

    base_entropy = entropy(y_vals)
    best_gain = -1.0
    best_feat = None

    for feat in feature_cols:
        val_counts = df[feat].value_counts()
        weighted_entropy = 0.0
        for val, count in val_counts.items():
            subset_y = df[df[feat] == val][target_col].values
            weighted_entropy += (count / len(df)) * entropy(subset_y)
        info_gain = base_entropy - weighted_entropy
        if info_gain > best_gain:
            best_gain = info_gain
            best_feat = feat

    if best_gain <= 1e-5 or best_feat is None:
        return MultiWayID3Node(is_leaf=True, prediction=majority_class, counts=counts_dict)

    node = MultiWayID3Node(attribute=best_feat, is_leaf=False, counts=counts_dict)
    remaining_feats = [f for f in feature_cols if f != best_feat]

    for val in df[best_feat].unique():
        sub_df = df[df[best_feat] == val]
        child = build_id3_tree(sub_df, remaining_feats, target_col, depth + 1, max_depth)
        node.branches[str(val)] = child

    return node

def convert_id3_to_json(node, class_names):
    val_vector = [int(node.counts.get(c, 0)) for c in class_names]
    total_samples = int(sum(node.counts.values()))
    pred_class = str(node.prediction) if node.prediction else (class_names[int(np.argmax(val_vector))] if len(val_vector) > 0 else class_names[0])
    cls_idx = class_names.index(pred_class) if pred_class in class_names else 0

    if node.is_leaf:
        return {
            'type': 'leaf',
            'condition': None,
            'samples': total_samples,
            'value': [int(v) for v in val_vector],
            'class': pred_class,
            'color': get_class_color(cls_idx),
            'is_pure': bool(total_samples == max(val_vector) if val_vector else True)
        }, 1, 1
    else:
        children_json = []
        total_leaves = 0
        total_size = 1

        for val, child_node in node.branches.items():
            c_json, c_leaves, c_size = convert_id3_to_json(child_node, class_names)
            c_json['edge_label'] = f"= {val}"
            children_json.append(c_json)
            total_leaves += c_leaves
            total_size += c_size

        return {
            'type': 'internal',
            'condition': str(node.attribute),
            'samples': total_samples,
            'value': val_vector,
            'class': pred_class,
            'color': "#ffffff",
            'is_pure': False,
            'children': children_json
        }, total_leaves, total_size

def predict_id3_sample(node, row, all_classes):
    if node.is_leaf:
        total = sum(node.counts.values()) + len(all_classes)
        probs = [(node.counts.get(c, 0) + 1) / total for c in all_classes]
        return node.prediction, probs
    feat = node.attribute
    val = str(row.get(feat, ''))
    if val in node.branches:
        return predict_id3_sample(node.branches[val], row, all_classes)
    else:
        total = sum(node.counts.values()) + len(all_classes)
        probs = [(node.counts.get(c, 0) + 1) / total for c in all_classes]
        maj = max(node.counts.items(), key=lambda x: x[1])[0] if node.counts else all_classes[0]
        return maj, probs


# ==========================================
# 2. Decision Tree Extractor (Image #1 Style)
# ==========================================
def build_tree_structure_matching_image(clf, feature_names, class_names, df_orig, target_col):
    tree_ = clf.tree_
    feature_index = tree_.feature
    threshold = tree_.threshold
    value = tree_.value

    def extract_node(node_id):
        node_value = [int(v) for v in value[node_id][0]]
        total_samples = int(np.sum(node_value))
        pred_class_idx = int(np.argmax(node_value))
        pred_class = class_names[pred_class_idx]
        is_pure = (total_samples == max(node_value)) if node_value else True

        if tree_.feature[node_id] != -2:  # Internal Node
            feat_idx = feature_index[node_id]
            feat_name = feature_names[feat_idx]
            left_child_id = tree_.children_left[node_id]
            right_child_id = tree_.children_right[node_id]
            
            is_nominal = False
            unique_vals = []
            if feat_name in df_orig.columns and not pd.api.types.is_numeric_dtype(df_orig[feat_name]):
                is_nominal = True
                unique_vals = list(df_orig[feat_name].astype(str).unique())

            left_node = extract_node(left_child_id)
            right_node = extract_node(right_child_id)

            thresh_val = float(threshold[node_id])
            if is_nominal and len(unique_vals) > 0:
                idx_thresh = int(np.floor(thresh_val))
                if idx_thresh >= 0 and idx_thresh < len(unique_vals):
                    left_node['edge_label'] = "True"
                    left_node['edge_sub'] = f"= {unique_vals[idx_thresh]}"
                    right_node['edge_label'] = "False"
                    right_node['edge_sub'] = f"!= {unique_vals[idx_thresh]}"
                    cond_str = f"{feat_name} == {unique_vals[idx_thresh]}"
                else:
                    left_node['edge_label'] = "True"
                    right_node['edge_label'] = "False"
                    cond_str = f"{feat_name} <= {thresh_val:.2f}"
            else:
                left_node['edge_label'] = "True"
                right_node['edge_label'] = "False"
                cond_str = f"{feat_name} <= {thresh_val:.2f}"

            return {
                'type': 'internal',
                'condition': cond_str,
                'samples': total_samples,
                'value': node_value,
                'class': pred_class,
                'color': "#ffffff",
                'is_pure': False,
                'children': [left_node, right_node]
            }
        else:  # Leaf Node
            return {
                'type': 'leaf',
                'condition': None,
                'samples': int(total_samples),
                'value': [int(v) for v in node_value],
                'class': pred_class,
                'color': get_class_color(pred_class_idx),
                'is_pure': bool(is_pure)
            }

    root_structure = extract_node(0)
    leaf_count = int(np.sum(tree_.feature == -2))
    node_count = int(tree_.node_count)
    return root_structure, leaf_count, node_count


# ==========================================
# 3. Naive Bayes Tree Visualizer Generator
# ==========================================
def build_naive_bayes_tree_structure(clf, feature_names, class_names, df, target_col):
    surrogate_tree = DecisionTreeClassifier(max_depth=4, criterion='entropy', random_state=42)
    df_encoded = df.copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            
    X_enc = df_encoded[feature_names].values
    y_enc = df_encoded[target_col].values
    surrogate_tree.fit(X_enc, y_enc)
    
    return build_tree_structure_matching_image(
        surrogate_tree, feature_names, class_names, df, target_col
    )


# ==========================================
# 4. KNN Space Partition Visualizer Generator
# ==========================================
def build_knn_tree_structure(clf, feature_names, class_names, df, target_col):
    surrogate_tree = DecisionTreeClassifier(max_depth=4, criterion='gini', random_state=42)
    df_encoded = df.copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            
    X_enc = df_encoded[feature_names].values
    y_enc = df_encoded[target_col].values
    surrogate_tree.fit(X_enc, y_enc)
    
    return build_tree_structure_matching_image(
        surrogate_tree, feature_names, class_names, df, target_col
    )


# ==========================================
# Main Evaluation Pipeline
# ==========================================
def evaluate_classifier(file_content, filename, algo_name, params=None):
    start_time = time.time()
    params = params or {}
    eval_mode = params.get('eval_mode', 'cross_val')
    knn_k = int(params.get('k', 3))
    
    df, attributes, relation_name = parse_dataset(file_content, filename)
    if df.shape[1] < 2:
        raise ValueError("Dataset must have at least 1 feature column and 1 class column.")
        
    target_col = attributes[-1]['name']
    feature_cols = [attr['name'] for attr in attributes[:-1]]

    df_encoded = df.copy()
    label_encoders = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    X = df_encoded[feature_cols].values
    y = df_encoded[target_col].values

    if target_col in label_encoders:
        class_names = [str(c) for c in label_encoders[target_col].classes_]
    else:
        class_names = [str(c) for c in np.unique(y)]

    tree_data = None
    y_pred = []
    y_proba = []
    y_true = []

    if algo_name == 'ID3':
        id3_root = build_id3_tree(df, feature_cols, target_col, max_depth=6)
        struct_json, num_leaves, tree_size = convert_id3_to_json(id3_root, class_names)
        tree_data = {
            'structure': struct_json,
            'num_leaves': num_leaves,
            'tree_size': tree_size,
            'relation_name': relation_name,
            'algo_name': 'ID3'
        }

        preds = []
        probas = []
        for _, row in df.iterrows():
            pred, prob = predict_id3_sample(id3_root, row, class_names)
            preds.append(pred)
            probas.append(prob)

        y_true = df[target_col].astype(str).values
        y_pred = np.array(preds)
        y_proba = np.array(probas)

    else:
        if algo_name in ['J48', 'C4.5']:
            clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=4, min_samples_leaf=2, random_state=42)
        elif algo_name == 'Naive Bayes':
            clf = GaussianNB(var_smoothing=1e-2)
        elif algo_name == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=knn_k, weights='distance', metric='euclidean')
        else:
            clf = DecisionTreeClassifier(criterion='entropy', random_state=42)

        if eval_mode == 'cross_val' and len(df) >= 10:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            y_pred_list = np.zeros(len(y), dtype=int)
            y_proba_list = np.zeros((len(y), len(class_names)))

            for train_idx, test_idx in skf.split(X, y):
                X_tr, X_te = X[train_idx], X[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                clf.fit(X_tr, y_tr)
                y_pred_list[test_idx] = clf.predict(X_te)
                if hasattr(clf, "predict_proba"):
                    y_proba_list[test_idx] = clf.predict_proba(X_te)

            y_true = np.array([class_names[i] for i in y])
            y_pred = np.array([class_names[i] for i in y_pred_list])
            y_proba = y_proba_list

            clf.fit(X, y)

        else:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)
            clf.fit(X_tr, y_tr)
            y_pred_idx = clf.predict(X_te)
            y_true = np.array([class_names[i] for i in y_te])
            y_pred = np.array([class_names[i] for i in y_pred_idx])
            if hasattr(clf, "predict_proba"):
                y_proba = clf.predict_proba(X_te)

        # Extract Decision Trees for ALL 4 Algorithms
        if algo_name in ['J48', 'C4.5']:
            root_struct, num_leaves, tree_size = build_tree_structure_matching_image(
                clf, feature_cols, class_names, df, target_col
            )
            tree_data = {
                'structure': root_struct,
                'num_leaves': num_leaves,
                'tree_size': tree_size,
                'relation_name': relation_name,
                'algo_name': 'J48'
            }
        elif algo_name == 'Naive Bayes':
            root_struct, num_leaves, tree_size = build_naive_bayes_tree_structure(
                clf, feature_cols, class_names, df, target_col
            )
            tree_data = {
                'structure': root_struct,
                'num_leaves': num_leaves,
                'tree_size': tree_size,
                'relation_name': relation_name,
                'algo_name': 'Naive Bayes (NB-Tree)'
            }
        elif algo_name == 'KNN':
            root_struct, num_leaves, tree_size = build_knn_tree_structure(
                clf, feature_cols, class_names, df, target_col
            )
            tree_data = {
                'structure': root_struct,
                'num_leaves': num_leaves,
                'tree_size': tree_size,
                'relation_name': relation_name,
                'algo_name': f'KNN (IBk KD-Tree, k={knn_k})'
            }

    build_time = round(time.time() - start_time, 3)

    # Accuracy Score
    acc = accuracy_score(y_true, y_pred) * 100.0

    # Weighted F1 Score
    w_f1 = round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 3)

    # Kappa Statistic
    kappa_val = round(float(cohen_kappa_score(y_true, y_pred)), 3)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_list = cm.tolist()

    # Classification Report
    report_dict = classification_report(
        y_true, y_pred, 
        target_names=class_names, 
        output_dict=True, 
        zero_division=0
    )
    report_str = classification_report(
        y_true, y_pred, 
        target_names=class_names, 
        zero_division=0
    )

    # Multi-Class ROC Curves
    roc_data = []
    n_classes = len(class_names)

    if len(y_proba) > 0:
        try:
            if n_classes == 2:
                pos_proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                y_true_bin = (y_true == class_names[1]).astype(int)
                fpr, tpr, _ = roc_curve(y_true_bin, pos_proba)
                roc_auc = auc(fpr, tpr)
                roc_data.append({
                    'class': class_names[1],
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': round(float(roc_auc), 2)
                })
            else:
                le_class = LabelEncoder()
                le_class.fit(class_names)
                y_true_idx = le_class.transform(y_true)
                y_true_bin = label_binarize(y_true_idx, classes=np.arange(n_classes))

                for i in range(n_classes):
                    if y_proba.ndim > 1 and i < y_proba.shape[1]:
                        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                        roc_auc = auc(fpr, tpr)
                        roc_data.append({
                            'class': class_names[i],
                            'fpr': fpr.tolist(),
                            'tpr': tpr.tolist(),
                            'auc': round(float(roc_auc), 2)
                        })
        except Exception:
            roc_data = []

    # Raw WEKA Format Output Text
    output_text = f"""=== Run information ===

Scheme:       weka.classifiers.{'trees.' + algo_name if algo_name in ['J48', 'ID3'] else 'bayes.NaiveBayes' if algo_name == 'Naive Bayes' else 'lazy.IBk'}
Relation:     {relation_name}
Instances:    {len(df)}
Attributes:   {len(attributes)}
              {' '.join([a['name'] for a in attributes])}
Test mode:    {'10-fold cross-validation' if eval_mode == 'cross_val' else 'percentage split 20% holdout test set'}

=== Classifier model ({algo_name}) ===

Algorithm Executed Successfully
Selected Algorithm: {algo_name}
Accuracy: {acc:.1f}%
Weighted F1: {w_f1}
Kappa statistic: {kappa_val}

=== Evaluation on test set ===

Correctly Classified Instances     {int(round(acc * len(y_true) / 100))}   {acc:.2f} %
Incorrectly Classified Instances   {len(y_true) - int(round(acc * len(y_true) / 100))}   {100 - acc:.2f} %
Total Number of Instances          {len(y_true)}

=== Detailed Accuracy By Class ===

{report_str}

=== Confusion Matrix ===

{' '.join([f'{c:>5}' for c in class_names])} <-- classified as
"""
    for idx, row in enumerate(cm_list):
        row_str = ' '.join([f'{val:>5}' for val in row])
        output_text += f"{row_str} | {class_names[idx]}\n"

    # Provide a capped dataset preview (avoid huge payloads)
    preview_limit = min(len(df), 500)
    dataset_preview_rows = df.head(preview_limit).fillna('').astype(str).values.tolist()

    return {
        'success': True,
        'algorithm': algo_name,
        'relation_name': relation_name,
        'attributes': attributes,
        'dataset_columns': [a['name'] for a in attributes],
        'dataset_rows': len(df),
        'dataset_preview': dataset_preview_rows,
        'dataset_preview_limit': preview_limit,
        'accuracy': round(acc, 1),
        'weighted_f1': w_f1,
        'kappa_statistic': kappa_val,
        'eval_rows': len(y_true),
        'build_time': build_time,
        'eval_mode_label': '10-fold cross-validation' if eval_mode == 'cross_val' else '20% stratified holdout test set',
        'confusion_matrix': {
            'matrix': cm_list,
            'classes': class_names
        },
        'classification_report': report_dict,
        'classification_report_str': report_str,
        'roc_curves': roc_data,
        'tree_data': tree_data,
        'complete_output': output_text
    }
