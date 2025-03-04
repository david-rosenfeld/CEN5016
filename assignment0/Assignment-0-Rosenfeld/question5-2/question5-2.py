import pandas as pd

# The CSV data has no headers.  These headers will be added after import.
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

# The names of the files to import.
datafiles = ["data/census-income.data.csv", "data/census-income.test.csv"]

# Initialize an array to hold dataframes for the data and test sets.
dataframes = []

# Load the CSV files.
for f in datafiles:
    # Read the file into a dataframe.  As noted, there are no headers.
    df = pd.read_csv(f, header=None)
    # Add the column headers.
    df.columns = column_headers
    
    # Data cleaning operations:

    # Remove duplicates.  According to the problem statement, there are duplicate
    # or conflicting instances in both the data and test sets.
    df.drop_duplicates(inplace=True)

    # Handle missing values using forward fill and backward fill.
    df = df.ffill().bfill()

    # Upon analyzying the data, the two values in the "income_50k_plus" column, after
    # import, look like this: [' - 50000.', ' 50000+.'].  I am replacing these values using
    # one-hot encoding so that less than 50k income is a 0, and 50k+ is a 1.
    
    # Strip whitespace
    df["income_50k_plus"] = df["income_50k_plus"].str.strip()
    # Map the existing values to 0 or 1.
    df["income_50k_plus"] = df["income_50k_plus"].map({"- 50000.": 0, "50000+.": 1})

    # Upon analyzing the data, the the values in the "race" column contain leading
    # and/or trailing spaces.  These will be removed.

    # Strip whitespace
    df["race"] = df["race"].str.strip()

    # Add this dataframe to the array
    dataframes.append(df)

# Assign the data frames as training or test data.
training_data = dataframes[0]
test_data = dataframes[1]

# Answer the questions in the problem statement:

# a) Based on the training data, how many people have an income of more than 50K per year?
training_50k_plus = training_data[training_data["income_50k_plus"] == 1].shape[0]
print(f"Training data, income 50K+: {training_50k_plus}")

# b) Based on the testing data, how many people have an income of more than 50K per year?
test_50k_plus = test_data[test_data["income_50k_plus"] == 1].shape[0]
print(f"Test data, income 50K+: {test_50k_plus}")

# c) Based on the testing data, how many people are Asian or Pacific Islander?
test_api = test_data[test_data["race"] == "Asian or Pacific Islander"].shape[0]
print(f"Test data, Asian or Pacific Islander: {test_api}")

# d) Based on the training data, what is the average age of people with more than 50K income per year?
# Filter the rows that have the higher income into a new dataframe
high_income = training_data[training_data["income_50k_plus"] == 1]
# Calculate the mean age.
mean_age = high_income["age"].mean()
print(f"Average age, 50K+ income: {mean_age}")

# e) Based on the testing data, what is the average age of people with more than 50K income per year?
# Filter the rows that have the higher income into a new dataframe
high_income = test_data[test_data["income_50k_plus"] == 1]
# Calculate the mean age.
mean_age = high_income["age"].mean()
print(f"Average age, 50K+ income: {mean_age}")
