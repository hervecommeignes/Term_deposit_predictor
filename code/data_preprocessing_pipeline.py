#!/usr/bin/env python3
"""
Data Preprocessing Pipeline
Transforms raw training and test data according to specified requirements:
- Remove: day, month, id columns
- Transform pdays: -1 -> 1000
- Remove rows with null fields
- Convert numeric fields to int32
- Convert string fields to categorical while preserving meaning
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset according to specified requirements.
    
    Args:
        df (pd.DataFrame): Raw dataset to preprocess
        
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    print(f"Starting preprocessing...")
    print(f"Original shape: {df.shape}")
    
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Step 0: Remove id column
    print("Step 0: Removing id column...")
    df_processed = df_processed.drop(columns=['id'])
    df_processed = df_processed.drop_duplicates()

    # Step 1: Remove specified columns
    print("Step 1: Removing day, month columns...")
    columns_to_remove = ['day', 'month']
    existing_columns_to_remove = [col for col in columns_to_remove if col in df_processed.columns]
    
    if existing_columns_to_remove:
        df_processed = df_processed.drop(columns=existing_columns_to_remove)
        print(f"  Removed columns: {existing_columns_to_remove}")
    else:
        print("  No specified columns found to remove")
    
    print(f"  Shape after removal: {df_processed.shape}")
    
    # Step 2: Transform pdays (-1 -> 1000)
    print("Step 2: Transforming pdays column...")
    if 'pdays' in df_processed.columns:
        # Count -1 values before transformation
        pdays_neg1_count = (df_processed['pdays'] == -1).sum()
        print(f"  Found {pdays_neg1_count} rows with pdays = -1")
        
        # Transform -1 to 1000
        df_processed['pdays'] = df_processed['pdays'].replace(-1, 1000)
        
        # Verify transformation
        pdays_neg1_after = (df_processed['pdays'] == -1).sum()
        pdays_1000_count = (df_processed['pdays'] == 1000).sum()
        print(f"  After transformation: {pdays_neg1_after} rows with pdays = -1, {pdays_1000_count} rows with pdays = 1000")
    else:
        print("  pdays column not found")
    
    # Step 3: Remove rows with null fields
    print("Step 3: Removing rows with null fields...")
    null_counts_before = df_processed.isnull().sum()
    rows_with_nulls_before = df_processed.isnull().any(axis=1).sum()
    
    print(f"  Rows with nulls before: {rows_with_nulls_before}")
    print(f"  Null counts per column:")
    for col, null_count in null_counts_before.items():
        if null_count > 0:
            print(f"    {col}: {null_count}")
    
    # Remove rows with any null values
    df_processed = df_processed.dropna()
    
    null_counts_after = df_processed.isnull().sum()
    rows_with_nulls_after = df_processed.isnull().any(axis=1).sum()
    
    print(f"  Rows with nulls after: {rows_with_nulls_after}")
    print(f"  Rows removed: {rows_with_nulls_before - rows_with_nulls_after}")
    print(f"  Shape after null removal: {df_processed.shape}")
    
    # Step 4: Convert numeric fields to int32
    print("Step 4: Converting numeric fields to int32...")
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude target variable if it exists (keep as int64 for binary classification)
    if 'y' in numeric_columns:
        numeric_columns.remove('y')
    
    print(f"  Numeric columns to convert: {numeric_columns}")
    
    for col in numeric_columns:
        try:
            # Convert to int32, handling any potential overflow
            df_processed[col] = df_processed[col].astype(np.int32)
            print(f"    {col}: converted to int32")
        except Exception as e:
            print(f"    {col}: conversion failed - {e}")
            # Try converting to float first, then int32
            try:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').astype(np.int32)
                print(f"    {col}: converted to int32 (after float conversion)")
            except Exception as e2:
                print(f"    {col}: final conversion failed - {e2}")
    
    # Step 5: Convert string fields to categorical while preserving meaning
    print("Step 5: Converting string fields to categorical...")
    string_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    print(f"  String columns to convert: {string_columns}")
    
    for col in string_columns:
        try:
            # Convert to categorical, preserving original values
            df_processed[col] = df_processed[col].astype('category')
            
            # Get unique values to show what was preserved
            unique_values = df_processed[col].cat.categories.tolist()
            print(f"    {col}: converted to categorical with {len(unique_values)} categories")
            print(f"      Categories: {unique_values}")
            
        except Exception as e:
            print(f"    {col}: conversion failed - {e}")
    
    # Final data type summary
    print("\nFinal data types:")
    print(df_processed.dtypes)
    
    print(f"\nPreprocessing completed!")
    print(f"Final shape: {df_processed.shape}")
    
    return df_processed

def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Preprocess a single dataset with the specified transformations.
    
    Args:
        input_path (str): Path to raw data file
        output_path (str): Path to save processed data
        
    Returns:
        pd.DataFrame: Processed dataset
    """
    print(f"Processing data...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print("-" * 60)
    
    # Load raw data
    df = pd.read_csv(input_path)
    print(f"Raw data shape: {df.shape}")
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Save processed data
    df_processed.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")
    
    # Summary
    print(f"\nDATA SUMMARY:")
    print(f"  Original: {df.shape} -> Processed: {df_processed.shape}")
    print(f"  Rows removed: {df.shape[0] - df_processed.shape[0]}")
    print(f"  Columns removed: {df.shape[1] - df_processed.shape[1]}")
    
    return df_processed

def main():
    """Main function to run the preprocessing pipeline."""
    
    print("="*80)
    print("DATA PREPROCESSING PIPELINE")
    print("="*80)
    
    # Define file paths
    train_path = "/Users/herve/Downloads/classif/data/train.csv"
    test_path = "/Users/herve/Downloads/classif/data/test.csv"
    output_train_path = "/Users/herve/Downloads/classif/data/train_processed.csv"
    output_test_path = "/Users/herve/Downloads/classif/data/test_processed.csv"
    
    # Process training data
    print("\n" + "="*50)
    print("PROCESSING TRAINING DATA")
    print("="*50)
    train_processed = preprocess(train_path, output_train_path)
    
    # Process test data
    print("\n" + "="*50)
    print("PROCESSING TEST DATA")
    print("="*50)
    test_processed = preprocess(test_path, output_test_path)
    
    # Final summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"Training data: {train_processed.shape}")
    print(f"Test data: {test_processed.shape}")
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()
