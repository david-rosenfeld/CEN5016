# Extend the existing code to generate depth-3 splits

# Compute the best attributes to split at depth 3
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
                depth_3_splits[v][best_sub_attr][vv] = None  # Pure node, no further split
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

# Convert depth-3 dictionary to a structured DataFrame
depth_3_flat = []
for k, v in decision_tree_depth_3[best_attr].items():
    row = {
        'Root Split Value': k,
        'Entropy': v['Entropy'],
        'Positives': v['Positives'],
        'Negatives': v['Negatives'],
        'Next Split': v['Next Split'],
        'Depth 3 Splits': v['Depth 3 Splits']
    }
    depth_3_flat.append(row)

# Convert to DataFrame
depth_3_df = pd.DataFrame(depth_3_flat)

# Display results
import ace_tools as tools
tools.display_dataframe_to_user(name="Decision Tree - Depth 3", dataframe=depth_3_df)
