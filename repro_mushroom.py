from weka_engine import parse_dataset, evaluate_classifier
import os
path = os.path.join('datasets', 'mushroom_full.arff')
with open(path, 'r', encoding='utf-8', errors='ignore') as f:
    data = f.read()
print('parsing...')
df, attrs, rel = parse_dataset(data, os.path.basename(path))
print('parsed', df.shape, 'cols', len(attrs), 'first', attrs[0]['name'], 'last', attrs[-1]['name'])
print(df.head(2).to_string())
print('evaluating...')
res = evaluate_classifier(data, os.path.basename(path), 'J48', {'k': 3})
print('success', res['dataset_rows'], len(res['dataset_preview']), res['dataset_columns'][:5], '...')
print('preview row example:', res['dataset_preview'][0][:10])
