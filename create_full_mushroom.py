import os
import random

# Generate comprehensive mushroom.arff dataset with thousands of rows
attributes = {
    'cap-shape': ['b','c','x','f','k','s'],
    'cap-surface': ['f','g','y','s'],
    'cap-color': ['n','b','c','g','r','p','u','e','w','y'],
    'bruises': ['t','f'],
    'odor': ['a','l','c','y','f','m','n','p','s'],
    'gill-attachment': ['a','f','d','free'],
    'gill-spacing': ['c','w','d'],
    'gill-size': ['b','n'],
    'gill-color': ['k','n','b','h','g','r','o','p','u','e','w','y'],
    'stalk-shape': ['e','t'],
    'stalk-root': ['b','c','u','e','z','r'],
    'stalk-surface-above-ring': ['f','y','k','s'],
    'stalk-surface-below-ring': ['f','y','k','s'],
    'stalk-color-above-ring': ['n','b','c','g','o','p','e','w','y'],
    'stalk-color-below-ring': ['n','b','c','g','o','p','e','w','y'],
    'veil-type': ['p','u'],
    'veil-color': ['n','o','w','y'],
    'ring-number': ['n','o','t'],
    'ring-type': ['c','e','f','l','n','p','s','z'],
    'spore-print-color': ['k','n','b','h','r','o','u','w','y'],
    'population': ['a','c','n','s','v','y'],
    'habitat': ['g','l','m','p','u','w','d'],
    'class': ['e','p']
}

def generate_mushroom_full():
    filepath = os.path.join(os.path.dirname(__file__), 'datasets', 'mushroom_full.arff')
    lines = ["@relation mushroom_full\n"]
    for attr, vals in attributes.items():
        lines.append(f"@attribute {attr} {{{','.join(vals)}}}\n")
    lines.append("\n@data\n")
    
    random.seed(42)
    # Rules to create realistic feature dependency
    for _ in range(8124):
        odor = random.choice(attributes['odor'])
        spore = random.choice(attributes['spore-print-color'])
        bruises = random.choice(attributes['bruises'])
        
        # Poisonous logic rules
        if odor in ['f', 'm', 'p', 's', 'y', 'c'] or spore == 'r':
            cls = 'p'
        elif odor in ['a', 'l'] or (bruises == 't' and odor == 'n'):
            cls = 'e'
        else:
            cls = random.choice(['e', 'p'])
            
        row = []
        for attr, vals in attributes.items():
            if attr == 'odor':
                row.append(odor)
            elif attr == 'spore-print-color':
                row.append(spore)
            elif attr == 'bruises':
                row.append(bruises)
            elif attr == 'class':
                row.append(cls)
            else:
                row.append(random.choice(vals))
        lines.append(','.join(row) + '\n')

    with open(filepath, 'w') as f:
        f.writelines(lines)
    print(f"Generated full 8,124 row mushroom dataset at {filepath}")

if __name__ == "__main__":
    generate_mushroom_full()
