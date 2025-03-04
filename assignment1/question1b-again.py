# Re-import required libraries
import numpy as np
import pandas as pd
import math


# Correcting dataset column names
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

# Compute dataset entropy
H_S = entropy(data['A'])

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
        # Compute IG for remaining attributes
        remaining_attrs = [attr for attr in attributes if attr != best_attr]
        ig_sub = {attr: information_gain(subset, attr, 'A') for attr in remaining_attrs}
        best_sub_attr = max(ig_sub, key=ig_sub.get)
        depth_2_splits[v] = best_sub_attr

# Construct decision trees
decision_tree_depth_1 = {best_attr: {v: {'Entropy': entropy(subset['A']), 'Positives': sum(subset['A']), 'Negatives': len(subset) - sum(subset['A'])} for v, subset in split_1.items()}}

decision_tree_depth_2 = {best_attr: {v: {'Entropy': entropy(subset['A']), 'Positives': sum(subset['A']), 'Negatives': len(subset) - sum(subset['A']), 'Next Split': depth_2_splits[v]} for v, subset in split_1.items()}}

# Display results
depth_1_df = pd.DataFrame.from_dict(decision_tree_depth_1[best_attr], orient='index')
depth_2_df = pd.DataFrame.from_dict(decision_tree_depth_2[best_attr], orient='index')

tools.display_dataframe_to_user(name="Decision Tree - Depth 1", dataframe=depth_1_df)
tools.display_dataframe_to_user(name="Decision Tree - Depth 2", dataframe=depth_2_df)
