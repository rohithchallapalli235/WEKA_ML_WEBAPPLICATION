import os

def build_mappings():
    return {
        'cap-shape': {
            'b': 'bell', 'c': 'conical', 'x': 'convex', 'f': 'flat', 'k': 'knobbed', 's': 'sunken'
        },
        'cap-surface': {
            'f': 'fibrous', 'g': 'grooves', 'y': 'scaly', 's': 'smooth'
        },
        'cap-color': {
            'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'r': 'green', 'p': 'pink',
            'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow'
        },
        'bruises': {
            't': 'true', 'f': 'false'
        },
        'odor': {
            'a': 'almond', 'l': 'anise', 'c': 'creosote', 'y': 'fishy', 'f': 'foul',
            'm': 'musty', 'n': 'none', 'p': 'pungent', 's': 'spicy'
        },
        'gill-attachment': {
            'a': 'attached', 'f': 'free', 'd': 'descending'
        },
        'gill-spacing': {
            'c': 'close', 'w': 'crowded', 'd': 'distant'
        },
        'gill-size': {
            'b': 'broad', 'n': 'narrow'
        },
        'gill-color': {
            'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate', 'g': 'gray',
            'r': 'green', 'o': 'orange', 'p': 'pink', 'u': 'purple', 'e': 'red',
            'w': 'white', 'y': 'yellow'
        },
        'stalk-shape': {
            'e': 'enlarging', 't': 'tapering'
        },
        'stalk-root': {
            'b': 'bulbous', 'c': 'club', 'u': 'cup', 'e': 'equal', 'z': 'rhizomorphs', 'r': 'rooted'
        },
        'stalk-surface-above-ring': {
            'f': 'fibrous', 'y': 'scaly', 'k': 'silky', 's': 'smooth'
        },
        'stalk-surface-below-ring': {
            'f': 'fibrous', 'y': 'scaly', 'k': 'silky', 's': 'smooth'
        },
        'stalk-color-above-ring': {
            'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'o': 'orange',
            'p': 'pink', 'e': 'red', 'w': 'white', 'y': 'yellow'
        },
        'stalk-color-below-ring': {
            'n': 'brown', 'b': 'buff', 'c': 'cinnamon', 'g': 'gray', 'o': 'orange',
            'p': 'pink', 'e': 'red', 'w': 'white', 'y': 'yellow'
        },
        'veil-type': {
            'p': 'partial', 'u': 'universal'
        },
        'veil-color': {
            'n': 'brown', 'o': 'orange', 'w': 'white', 'y': 'yellow'
        },
        'ring-number': {
            'n': 'none', 'o': 'one', 't': 'two'
        },
        'ring-type': {
            'c': 'cobwebby', 'e': 'evanescent', 'f': 'flaring', 'l': 'large',
            'n': 'none', 'p': 'pendant', 's': 'sheathing', 'z': 'zone'
        },
        'spore-print-color': {
            'k': 'black', 'n': 'brown', 'b': 'buff', 'h': 'chocolate', 'r': 'green',
            'o': 'orange', 'u': 'purple', 'w': 'white', 'y': 'yellow'
        },
        'population': {
            'a': 'abundant', 'c': 'clustered', 'n': 'numerous', 's': 'scattered',
            'v': 'several', 'y': 'solitary'
        },
        'habitat': {
            'g': 'grasses', 'l': 'leaves', 'm': 'meadows', 'p': 'paths',
            'u': 'urban', 'w': 'waste', 'd': 'woods'
        },
        'class': {
            'e': 'edible', 'p': 'poisonous'
        }
    }


def transform_dataset_file(path, mappings):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    attr_order = []
    new_lines = []
    in_data = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue
        if stripped.lower().startswith('@attribute'):
            parts = stripped.split(maxsplit=2)
            if len(parts) >= 3:
                attr_name = parts[1]
                if attr_name in mappings:
                    values = mappings[attr_name]
                    mapped_values = [values[val] if val in values else val for val in re.findall(r"\w+|'[^']+'|\"[^\"]+\"", parts[2]) if val not in ['{', '}', ',']]
                    new_line = f"@attribute {attr_name} {{{','.join(mapped_values)}}}\n"
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
            attr_order.append(attr_name)
            continue
        if stripped.lower().startswith('@data'):
            in_data = True
            new_lines.append(line)
            continue
        if in_data and stripped and not stripped.startswith('%'):
            values = [v.strip() for v in stripped.split(',')]
            if len(values) == len(attr_order):
                mapped_row = []
                for attr_name, value in zip(attr_order, values):
                    mapped_dict = mappings.get(attr_name, {})
                    mapped_row.append(mapped_dict.get(value, value))
                new_lines.append(','.join(mapped_row) + '\n')
            else:
                new_lines.append(line)
            continue
        new_lines.append(line)

    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


if __name__ == '__main__':
    import re

    workspace = os.path.dirname(__file__)
    dataset_files = ['datasets/mushroom.arff', 'datasets/mushroom_full.arff']
    mappings = build_mappings()

    for ds in dataset_files:
        path = os.path.join(workspace, ds)
        print('Transforming', path)
        transform_dataset_file(path, mappings)

    # Replace generator code lists in create_full_mushroom.py
    generator_path = os.path.join(workspace, 'create_full_mushroom.py')
    with open(generator_path, 'r', encoding='utf-8') as f:
        content = f.read()

    attr_lines = []
    for attr_name, mapping in mappings.items():
        words = [mapping[k] for k in sorted(mapping.keys(), key=lambda x: list(mapping.keys()).index(x))]
        attr_lines.append(f"    '{attr_name}': {words},")

    start = content.index('attributes = {')
    end = content.index('\n}\n', start) + 2
    new_content = content[:start] + 'attributes = {\n' + '\n'.join(attr_lines) + '\n}\n\n' + content[end:]

    with open(generator_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print('Updated create_full_mushroom.py generator value lists')
