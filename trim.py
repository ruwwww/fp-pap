import pandas as pd

# 1. Load your original dataset
input_file = "data_train_engineered.csv"
output_file = "data_train_trimmed.csv"
df = pd.read_csv(input_file)

# 2. Find the position of US_rate and slice the columns
target_column = "US_rate"

if target_column in df.columns:
    # Get the index position of US_rate
    column_index = df.columns.get_loc(target_column)

    # Slice the dataframe to keep everything from start up to target column
    df_trimmed = df.iloc[:, : column_index + 1]

    # 3. Save the modified dataframe to a new CSV file
    df_trimmed.to_csv(output_file, index=False)
    print(f"Success! Saved columns up to {target_column} into '{output_file}'.")
else:
    print(f"Error: Column '{target_column}' was not found in the CSV.")