import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_and_preprocess_data(filepath):
    print("--- Step 1: Loading Data ---")
    df = pd.read_csv(filepath)
    print(f"Data loaded. Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    
    # Keep a copy of the original for later analysis
    df_original = df.copy()

    print("\n--- Step 2: Feature Engineering ---")
    
    # 1. Arrival Month
    # Convert date.of.reservation to datetime
    print("Creating 'Arrival Month'...")
    # Use errors='coerce' to handle invalid dates (e.g. 2018-02-29) and format='mixed' to handle variations
    df['date.of.reservation'] = pd.to_datetime(df['date.of.reservation'], errors='coerce')
    
    # Check for dropped dates
    n_dropped = df['date.of.reservation'].isna().sum()
    if n_dropped > 0:
        print(f"Warning: {n_dropped} rows have invalid dates and will be dropped.")
        df = df.dropna(subset=['date.of.reservation'])
    
    # Add lead time (days) to get arrival date
    df['arrival_date'] = df['date.of.reservation'] + pd.to_timedelta(df['lead.time'], unit='D')
    df['Arrival_Month'] = df['arrival_date'].dt.month
    
    # 2. Total Guests
    print("Creating 'Total Guests'...")
    df['Total_Guests'] = df['number.of.adults'] + df['number.of.children']
    
    # 3. Is_Family
    print("Creating 'Is_Family'...")
    df['Is_Family'] = (df['number.of.children'] > 0).astype(int)
    
    # 4. Total Nights
    print("Creating 'Total Nights'...")
    df['Total_Nights'] = df['number.of.weekend.nights'] + df['number.of.week.nights']
    
    # 5. Cancellation Ratio (Smoothing)
    print("Creating 'Cancellation Ratio'...")
    # Formula: (P_C + 1) / (P_not_C + P_C + 2)
    df['Cancellation_Ratio'] = (df['P.C'] + 1) / (df['P.not.C'] + df['P.C'] + 2)
    
    # 6. Price per Person
    print("Creating 'Price per Person'...")
    # Avoid division by zero if Total_Guests is 0 (unlikely but good practice)
    df['Price_per_Person'] = df.apply(lambda row: row['average.price'] / row['Total_Guests'] if row['Total_Guests'] > 0 else 0, axis=1)

    print("\n--- Step 3: Feature Selection (Dropping columns) ---")
    # Columns to drop as per notes
    cols_to_drop = ['Booking_ID', 'date.of.reservation', 'arrival_date', 'booking.status'] 
    # Note: We keep booking.status in a separate variable for evaluation, but drop from X
    
    # Extract target/labels for evaluation later
    labels_true = df['booking.status']
    
    # Drop unused columns from the feature set
    X = df.drop(columns=cols_to_drop)
    print(f"Dropped columns: {cols_to_drop}")
    print(f"Remaining columns: {X.columns.tolist()}")

    print("\n--- Step 4: Preprocessing (One-Hot Encoding & Scaling) ---")
    
    # Identify categorical and numerical columns
    categorical_cols = ['type.of.meal', 'room.type', 'market.segment.type']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    
    print(f"Categorical columns to encode: {categorical_cols}")
    print(f"Numerical columns to scale: {numerical_cols}")
    
    # Define transformers
    # OneHotEncoder: handle_unknown='ignore' to be safe, though dataset is fixed here
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    numerical_transformer = StandardScaler()
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after encoding
    # OneHotEncoder feature names
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_feature_names = numerical_cols + list(ohe_feature_names)
    
    # Convert back to DataFrame for easier handling
    X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)  # type: ignore
    
    print(f"Preprocessing complete. Final feature matrix shape: {X_processed_df.shape}")
    
    return X_processed_df, labels_true, df_original

if __name__ == "__main__":
    # Test run
    import os
    # Assuming the script is run from the python folder or root, adjust path
    # Try to find the csv relative to the script location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'project_cluster.csv')
    
    if os.path.exists(csv_path):
        load_and_preprocess_data(csv_path)
    else:
        print(f"File not found at {csv_path}")
