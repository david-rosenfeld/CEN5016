# Train decision trees for different depths and compute both training and test accuracy
depth_accuracies = []

for depth in range(2, 11):
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    depth_accuracies.append({"Depth": depth, "Training Accuracy": train_acc, "Test Accuracy": test_acc})

# Convert results to DataFrame and display
depth_accuracies_df = pd.DataFrame(depth_accuracies)

import ace_tools as tools
tools.display_dataframe_to_user(name="Decision Tree Training vs. Test Accuracies", dataframe=depth_accuracies_df)


# The two lines below do the same thing in different ways.
# They each find the Depth associated with the largest value for Training Accuracy.

best_depth = max(depth_accuracies, key=lambda x: x["Training Accuracy"])["Depth"]

