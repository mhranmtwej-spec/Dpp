import os
import pandas as pd

def get_cleaned_df():
    # --- Configuration ---
    # --- Path Management ---
    destination = "../data/raw"
    dataset_name = "nyc-yellow-taxi-trip-records-january-2024"
    dataset_dir = os.path.join(destination, dataset_name)

    # Dynamically identify the cleaned file
    # This ensures we always pick the '_cleaned' version if it exists
    try:
        files = os.listdir(dataset_dir)
        cleaned_files = [f for f in files if f.endswith('_cleaned.csv')]
        
        if not cleaned_files:
            raise FileNotFoundError("No cleaned dataset found. Please run the cleaning notebook first.")
        
        cleaned_csv_path = os.path.join(dataset_dir, cleaned_files[0])
        print(f" Loading Dataset: {cleaned_csv_path}")
        
        # --- Data Loading ---
        df = pd.read_csv(cleaned_csv_path)
        
        # Success Message
        print(f" Dataset Loaded Successfully! Total Rows: {len(df):,}")

        _set_types(df=df)

    except Exception as e:
        print(f"❌ Error: {e}")
    
    

    return df 



def _set_types(df):

    """
    # NYC Taxi January 2024: Technical Data Schema
    ## Phase 1.4: Strict Type Enforcement

    ### 1. Rationale for Schema Optimization
    With approximately 2.6 million records, maintaining the default object types is inefficient. By enforcing strict types, we:
    * **Prevent Precision Loss:** Financial columns require `float64` for accurate aggregation.
    * **Enable Time-Series Analysis:** Converting to `datetime64[ns]` allows for vectorized extractions (Hour, Day of Week, Weekday vs. Weekend).
    * **Optimize Memory Footprint:** Categorizing flags and using explicit integers reduces the RAM load by up to 40%.

    ### 2. Final Data Type Map

    | Logic Group | Feature Name | Data Type |
    | :--- | :--- | :--- |
    | **Temporal** | `tpep_pickup_datetime`, `tpep_dropoff_datetime` | `datetime64[ns]` |
    | **Numerical (Discrete)** | `VendorID`, `passenger_count`, `RatecodeID`, `PULocationID`, `DOLocationID`, `payment_type` | `int64` |
    | **Numerical (Continuous)**| `trip_distance`, `fare_amount`, `extra`, `mta_tax`, `tip_amount`, `tolls_amount`, `improvement_surcharge`, `total_amount`, `congestion_surcharge`, `Airport_fee`, `fare_per_mile`, `trip_duration` | `float64` |
    | **Categorical** | `store_and_fwd_flag` | `category` |

    ---

    ### 3. Structural Integrity Check
    After enforcement, the `df.info()` should reflect a streamlined memory usage, and the `Correlation Matrix` will be more computationally efficient as all features are now in their optimal mathematical format.
    """

    # 1. Enforce Datetime Types
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    # 2. Enforce Integer Types (Discrete identifiers)
    int_features = [
        'VendorID', 'passenger_count', 'RatecodeID', 
        'PULocationID', 'DOLocationID', 'payment_type'
    ]
    df[int_features] = df[int_features].astype('int64')

    # 3. Enforce Float Types (Continuous financial/distance metrics)
    float_features = [
        'trip_distance', 'fare_amount', 'extra', 'mta_tax', 
        'tip_amount', 'tolls_amount', 'improvement_surcharge', 
        'total_amount', 'congestion_surcharge', 'Airport_fee',
        'fare_per_mile', 'trip_duration'
    ]
    df[float_features] = df[float_features].astype('float64')

    # 4. Enforce Categorical Type (Efficiency for flags)
    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].astype('category')

    # Verification
    print("Type enforcement complete. New Memory Usage:")
    print(f"{df.memory_usage().sum() / 1024**2:.2f} MB")


