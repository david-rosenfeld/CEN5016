# Re-import necessary libraries since execution state was reset
import numpy as np
import pandas as pd
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# Reload and clean the dataset as previously described
column_headers = [
    "age",
    "class_of_worker",
    "detailed_industry_recode",
    "detailed_occupation_recode",
    "education",
    "wage_per_hour",
    "enroll_in_edu_inst_last_wk",
    "marital_stat",
    "major_industry_code",
    "major_occupation_code",
    "race",
    "hispanic_origin",
    "sex",
    "member_of_labor_union",
    "reason_for_unemployment",
    "full_or_part_time_employment_stat",
    "capital_gains",
    "capital_losses",
    "dividends_from_stocks",
    "tax_filer_stat",
    "region_of_previous_residence",
    "state_of_previous_residence",
    "detailed_household_and_family_stat",
    "detailed_household_summary_in_household",
    "unknown_value",
    "migration_code_change_in_msa",
    "migration_code_change_in_reg",
    "migration_code_move_within_reg",
    "live_in_this_house_1_year_ago",
    "migration_prev_res_in_sunbelt",
    "num_persons_worked_for_employer",
    "family_members_under_18",
    "country_of_birth_father",
    "country_of_birth_mother",
    "country_of_birth_self",
    "citizenship",
    "own_business_or_self_employed",
    "fill_inc_questionnaire_for_veterans_admin",
    "veterans_benefits",
    "weeks_worked",
    "year",
    "income_50k_plus"
]

# Simulated file paths (Replace with actual file paths when running locally)
datafiles = ["data/census-income.data.csv", "data/census-income.test.csv"]
dataframes = []

# Load and clean data
for f in datafiles:
    df = pd.read_csv(f, header=None)
    df.columns = column_headers
    df = df.ffill().bfill()
    df["income_50k_plus"] = df["income_50k_plus"].str.strip().map({"- 50000.": 0, "50000+.": 1})
    df["race"] = df["race"].str.strip()
    dataframes.append(df)

# Assign to training and test sets
training_data, test_data = dataframes

# Define features and target variable
target_col = "income_50k_plus"
feature_cols = [col for col in training_data.columns if col != target_col]

# Identify categorical and numerical features
categorical_cols = training_data.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = training_data.select_dtypes(include=["int64", "float64"]).columns.tolist()

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
encoded_train = encoder.fit_transform(training_data[categorical_cols])
encoded_test = encoder.transform(test_data[categorical_cols])

# Convert to DataFrame and concatenate with numerical features
encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_cols))
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_cols))

X_train = pd.concat([training_data[numerical_cols].reset_index(drop=True), encoded_train_df], axis=1)
X_test = pd.concat([test_data[numerical_cols].reset_index(drop=True), encoded_test_df], axis=1)

y_train = training_data[target_col]
y_test = test_data[target_col]

# Train decision trees for different depths and store accuracies
depth_accuracies = []
for depth in range(2, 11):
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    depth_accuracies.append({"Depth": depth, "Training Accuracy": train_acc})

# Convert results to DataFrame and display
depth_accuracies_df = pd.DataFrame(depth_accuracies)
import ace_tools as tools
tools.display_dataframe_to_user(name="Decision Tree Training Accuracies", dataframe=depth_accuracies_df)
