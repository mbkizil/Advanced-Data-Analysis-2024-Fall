
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    # ---------------------- #
    # 1. Import the Dataset  #
    # ---------------------- #
    # Update 'anes_data.csv' to the actual path/filename of your dataset
    file_path = 'anes.csv'
    
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print("Data successfully loaded!\n")

    print("Initial DataFrame shape:", df.shape)
    
    # Display the first few rows
    print("First 5 rows of the raw data:")
    print(df.head())
    print("\nData info (column types, non-null counts):")
    print(df.info())
    
    # ------------------------------------------- #
    # 2. Basic Data Cleaning and Preprocessing    #
    # ------------------------------------------- #
    
    # A. Identify special missing value codes (examples: -9, -8, -1; adjust as needed)
    #    These codes often appear in ANES data for "Don't know," "Refused," or other non-responses.
    special_missing_codes = [-9, -8, -7, -6, -5, -4, -3, -2, -1]
    
    # Replace special codes with NaN
    print("\nReplacing special missing value codes with NaN...")
    df.replace(special_missing_codes, np.nan, inplace=True)
    
    # B. Look at missing data summary
    print("Missing values per column after replacement:")
    print(df.isnull().sum())
    
    # C. (Optional) Drop columns with too many missing values (threshold is an example, tune as needed)
    # If you have columns that are almost entirely missing, you might decide to drop them.
    # threshold = 0.8  # 80% missing
    # columns_to_drop = [col for col in df.columns 
    #                    if df[col].isnull().mean() > threshold]
    # if columns_to_drop:
    #     print(f"\nDropping columns with > {threshold*100}% missing values:")
    #     print(columns_to_drop)
    #     df.drop(columns=columns_to_drop, axis=1, inplace=True)
    #     print("New DataFrame shape:", df.shape)
    
    # D. (Optional) Drop rows with too many missing values or impute
    # For demonstration, let's just drop rows with any missing values in key columns
    # key_columns = ['vote_choice', 'age', 'pid']  # example only; adapt to your data
    # before_drop = df.shape[0]
    # df.dropna(subset=key_columns, inplace=True)
    # after_drop = df.shape[0]
    # print(f"\nDropped {before_drop - after_drop} rows due to missing in {key_columns}.")
    # print("New DataFrame shape:", df.shape)
    
    # ------------------------------------------ #
    # 3. Exploratory Data Analysis (EDA)         #
    # ------------------------------------------ #
    
    # A. Summary Statistics
    print("\nBasic statistical summary (numerical columns):")
    print(df.describe())
    
    # B. Distribution of Key Variables
    # Adjust these variable names to match actual columns of interest
    variables_of_interest = ['age', 'income', 'ideology', 'vote_choice', 'VCF0705', 'VCF0706']
    existing_vars = [var for var in variables_of_interest if var in df.columns]
    
    if existing_vars:
        print(f"\nPlotting histograms for the following variables: {existing_vars}")
        for var in existing_vars:
            plt.figure(figsize=(6, 4))
            sns.histplot(data=df, x=var, kde=True, color='blue')
            plt.title(f"Distribution of {var}")
            plt.tight_layout()
            plt.show()
    else:
        print("\nNo matching variables found for distribution plotting. Check your column names.")
    
    # C. Correlation Analysis
    # Select only numerical columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        print("\nCorrelation matrix for numeric columns:")
        print(corr_matrix)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, square=True)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()
    else:
        print("\nNot enough numeric columns to compute a correlation matrix.")
    
    # D. Pairwise Relationships
    # (Be cautious with very large data sets; a pairplot can be slow or cluttered.)
    sample_size = 500  # Adjust as needed; large data can crash pairplot
    if df.shape[0] > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df.copy()
    
    # Only plotting a subset of numerical columns for clarity
    subset_cols = numeric_cols[:4]  # limit to first 4 numeric columns
    if len(subset_cols) > 1:
        print(f"\nCreating pairplot for a subset of numeric columns: {subset_cols}")
        sns.pairplot(df_sample[subset_cols].dropna())
        plt.show()
    
    # E. Categorical Analysis (Example)
    # If 'vote_choice' or 'pid' (party ID) is in the dataset, let's see a countplot.
    cat_variable = 'vote_choice'  # example only; adapt to your data
    if cat_variable in df.columns and df[cat_variable].dtype == 'O':
        print(f"\nCount of responses by '{cat_variable}':")
        print(df[cat_variable].value_counts(dropna=False))
        
        plt.figure(figsize=(6, 4))
        sns.countplot(x=df[cat_variable], palette='viridis')
        plt.title(f"Count Plot of {cat_variable}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    # ---------------------------------- #
    # End of Basic Preprocessing & EDA   #
    # ---------------------------------- #
    
    print("\n=== Preprocessing and EDA Complete ===")
    print("You can proceed with further analysis, e.g., advanced ML or deep learning.")


if __name__ == "__main__":
    main()