# Re-import required libraries since execution state was reset
import numpy as np
import pandas as pd
import math

# Define the dataset again
data = pd.DataFrame([
    [1, 1, 0, 0, 1, 1],
    [1, 1, 1, 0, 1, 1],
    [0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 1, 0],
    [0, 1, 1, 0, 0, 1],
    [0, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 1],
    [1, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 1],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 1],
], columns=['Early', 'Finished HMK', 'Senior', 'Likes Coffee', 'Liked The Last Jedi', 'A'])

# Define entropy function
def entropy(y):
    total = len(y)
    if total == 0:
        return 0
    p1 = sum(y) / total  # Proportion of class A = 1
    p0 = 1 - p1  # Proportion of class A = 0
    if p1 == 0 or p0 == 0:
        return 0
    return - (p1 * math.log2(p1) + p0 * math.log2(p0))

# Compute dataset entropy
H_S = entropy(data['A'])

# Compute information gain for each attribute
def information_gain(data, attribute, target):
    values = data[attribute].unique()
    weighted_entropy = 0
    for v in values:
        subset = data[data[attribute] == v][target]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)
    return H_S - weighted_entropy

# Compute IG for each attribute
attributes = ['Early', 'Finished HMK', 'Senior', 'Likes Coffee', 'Liked The Last Jedi']
ig_values = {attr: information_gain(data, attr, 'A') for attr in attributes}

# Select the best attribute for depth 1 split
best_attr = max(ig_values, key=ig_values.get)

# Split data based on best attribute
split_1 = {v: data[data[best_attr] == v] for v in data[best_attr].unique()}

# Compute second level splits for depth 2
depth_2_splits = {}
for v, subset in split_1.items():
    if len(subset['A'].unique()) == 1:
        depth_2_splits[v] = None  # Pure node, no further split
    else:
        remaining_attrs = [attr for attr in attributes if attr != best_attr]
        ig_sub = {attr: information_gain(subset, attr, 'A') for attr in remaining_attrs}
        best_sub_attr = max(ig_sub, key=ig_sub.get)
        depth_2_splits[v] = best_sub_attr

# Compute third level splits for depth 3
depth_3_splits = {}
for v, subset in split_1.items():
    if len(subset['A'].unique()) == 1:
        depth_3_splits[v] = None  # Pure node, no further split
    else:
        remaining_attrs = [attr for attr in attributes if attr != best_attr]
        ig_sub = {attr: information_gain(subset, attr, 'A') for attr in remaining_attrs}
        best_sub_attr = max(ig_sub, key=ig_sub.get)

        split_2 = {vv: subset[subset[best_sub_attr] == vv] for vv in subset[best_sub_attr].unique()}
        
        depth_3_splits[v] = {best_sub_attr: {}}
        
        for vv, subset_2 in split_2.items():
            if len(subset_2['A'].unique()) == 1:
                depth_3_splits[v][best_sub_attr][vv] = None  # Pure node
            else:
                remaining_attrs_2 = [attr for attr in remaining_attrs if attr != best_sub_attr]
                ig_sub_2 = {attr: information_gain(subset_2, attr, 'A') for attr in remaining_attrs_2}
                best_sub_attr_2 = max(ig_sub_2, key=ig_sub_2.get)
                depth_3_splits[v][best_sub_attr][vv] = best_sub_attr_2

# Construct decision tree for depth 3
decision_tree_depth_3 = {best_attr: {}}
for v, subset in split_1.items():
    decision_tree_depth_3[best_attr][v] = {
        'Entropy': entropy(subset['A']), 
        'Positives': sum(subset['A']), 
        'Negatives': len(subset) - sum(subset['A']), 
        'Next Split': depth_2_splits[v],
        'Depth 3 Splits': depth_3_splits[v] if depth_3_splits[v] is not None else "Leaf Node"
    }

# Display results
import ace_tools as tools

depth_3_df = pd.DataFrame.from_dict(decision_tree_depth_3[best_attr], orient='index')
tools.display_dataframe_to_user(name="Decision Tree - Depth 3", dataframe=depth_3_df)
