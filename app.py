import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

# --- Configuration ---
ARTIFACTS_DIR = 'model_artifacts' 

# --- Load Artifacts ---
@st.cache_resource 
def load_artifacts():
    """Loads the pre-trained model, preprocessor, and other necessary artifacts."""
    model_path = os.path.join(ARTIFACTS_DIR, 'best_voting_model.pkl')
    preprocessor_path = os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl')
    features_before_path = os.path.join(ARTIFACTS_DIR, 'feature_names_before_preprocessing.pkl')
    train_means_path = os.path.join(ARTIFACTS_DIR, 'train_numerical_feature_means.pkl')
    meteo_cols_path = os.path.join(ARTIFACTS_DIR, 'meteo_cols_used.pkl')
    target_col_path = os.path.join(ARTIFACTS_DIR, 'target_col_name.pkl')


    required_files = [model_path, preprocessor_path, features_before_path, train_means_path, meteo_cols_path, target_col_path]
    if not all(os.path.exists(p) for p in required_files):
        missing_files = [p for p in required_files if not os.path.exists(p)]
        st.error(f"Error: One or more artifact files are missing from the '{ARTIFACTS_DIR}' directory: {missing_files}. "
                 "Please run the training script first to generate these files.")
        return None
    
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        feature_names_before_preprocessing = joblib.load(features_before_path)
        train_numerical_feature_means = joblib.load(train_means_path)
        meteo_cols_used = joblib.load(meteo_cols_path)
        target_col_name = joblib.load(target_col_path)
        
        return {
            "model": model,
            "preprocessor": preprocessor,
            "feature_names_before": feature_names_before_preprocessing,
            "train_numerical_means": train_numerical_feature_means,
            "meteo_cols": meteo_cols_used,
            "target_col": target_col_name
        }
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None

# --- Helper function to get season ---
def get_season_for_input(month_num):
    if month_num in [12, 1, 2]: return 'Winter'
    if month_num in [3, 4, 5]: return 'Spring'
    if month_num in [6, 7, 8]: return 'Summer'
    return 'Autumn'

# --- Streamlit App UI and Logic ---
st.set_page_config(layout="wide")
st.title("Dengue Case Prediction (7 Days Ahead)")
st.markdown("""
This app predicts the number of dengue cases 7 days from a specified 'current date'.
It uses a pre-trained XGBoost model. Provide current data and data from 7 days ago.
""")

artifacts = load_artifacts()

if artifacts:
    st.sidebar.success("Model and artifacts loaded successfully!")

    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    feature_names_before_preprocessing = artifacts["feature_names_before"]
    train_numerical_feature_means = artifacts["train_numerical_means"]
    METEO_COLS_USED = artifacts["meteo_cols"]
    TARGET_COL_NAME = artifacts["target_col"] # Loaded from artifact

    st.sidebar.header("Input Data for Prediction")
    current_input_date = st.sidebar.date_input("Current Date for Input Data", datetime.today())
    
    st.sidebar.subheader("Today's Data (Day D)")
    # Use TARGET_COL_NAME for default value lookup
    current_dengue_cases_default = int(train_numerical_feature_means.get(TARGET_COL_NAME, 50))
    current_dengue_cases = st.sidebar.number_input(f"{TARGET_COL_NAME.replace('_', ' ').title()} (Today)", min_value=0, value=current_dengue_cases_default)
    
    current_meteo_inputs = {}
    for col in METEO_COLS_USED:
        default_val = train_numerical_feature_means.get(col, 20.0)
        current_meteo_inputs[col] = st.sidebar.number_input(f"{col.upper()} (Today)", value=float(default_val), format="%.2f")

    st.sidebar.subheader("Data from 7 Days Ago (Day D-7)")
    lag7_dengue_cases_default = int(train_numerical_feature_means.get(f"{TARGET_COL_NAME}_lag_7", 40))
    lag7_dengue_cases = st.sidebar.number_input(f"{TARGET_COL_NAME.replace('_', ' ').title()} (7 Days Ago)", min_value=0, value=lag7_dengue_cases_default)
    
    lag7_meteo_inputs = {}
    for col in METEO_COLS_USED:
        default_val = train_numerical_feature_means.get(f"{col}_lag_7", 18.0)
        lag7_meteo_inputs[f"{col}_lag_7"] = st.sidebar.number_input(f"{col.upper()} (7 Days Ago)", value=float(default_val), format="%.2f")

    if st.sidebar.button("Predict Dengue Cases for 7 Days Ahead"):
        input_data_dict = {}

        for f_name in feature_names_before_preprocessing:
            if f_name in train_numerical_feature_means:
                input_data_dict[f_name] = train_numerical_feature_means[f_name]
            else:
                input_data_dict[f_name] = np.nan 

        input_data_dict['month'] = current_input_date.month
        input_data_dict['year'] = current_input_date.year
        input_data_dict['day_of_week'] = current_input_date.strftime('%A')
        input_data_dict['season'] = get_season_for_input(current_input_date.month)

        input_data_dict[TARGET_COL_NAME] = float(current_dengue_cases)
        for col_meteo, val_meteo in current_meteo_inputs.items():
            if col_meteo in input_data_dict:
                 input_data_dict[col_meteo] = float(val_meteo)

        input_data_dict[f'{TARGET_COL_NAME}_lag_7'] = float(lag7_dengue_cases)
        for col_meteo_lag7, val_meteo_lag7 in lag7_meteo_inputs.items():
             if col_meteo_lag7 in input_data_dict: 
                input_data_dict[col_meteo_lag7] = float(val_meteo_lag7)
        
        input_df = pd.DataFrame([input_data_dict], columns=feature_names_before_preprocessing)
        
        if input_df.isnull().any().any():
            st.warning("Some features were not directly provided or imputed initially. "
                       "Double-checking imputation for remaining NaNs using training means where possible.")
            for col_idx, col_name in enumerate(input_df.columns): # Iterate by name for clarity
                if input_df[col_name].isnull().any(): # Check if this specific column has NaN
                    if col_name in train_numerical_feature_means:
                        input_df[col_name] = input_df[col_name].fillna(train_numerical_feature_means[col_name])
                    else: # Fallback for categoricals (should be set) or unexpected NaNs
                        st.error(f"Feature '{col_name}' is still NaN after initial imputation and has no mean value. Prediction might be inaccurate. Filling with 0 as last resort.")
                        input_df[col_name] = input_df[col_name].fillna(0)


        st.subheader("Prediction Results")
        
        num_imputed_features = sum(1 for f_name in feature_names_before_preprocessing 
                                   if f_name not in current_meteo_inputs and 
                                      f_name not in lag7_meteo_inputs and    
                                      f_name != TARGET_COL_NAME and               
                                      f_name != f'{TARGET_COL_NAME}_lag_7' and    
                                      f_name not in ['month', 'year', 'day_of_week', 'season'] and 
                                      f_name in train_numerical_feature_means) 

        if num_imputed_features > 0:
            st.info(f"""
                **Note on Input Simplification:** For this prediction, {num_imputed_features} features 
                (e.g., lags like `_lag_14`, `_lag_30`, and all rolling statistics like `_roll_mean_7`) 
                were not directly requested in the UI. These have been automatically filled using their 
                average values from the model's training data.
            """)
        
        try:
            input_processed = preprocessor.transform(input_df)
            prediction = model.predict(input_processed)
            predicted_cases = max(0, int(round(prediction[0]))) 

            prediction_date = current_input_date + timedelta(days=7)
            st.success(f"Predicted {TARGET_COL_NAME.replace('_', ' ').title()} for {prediction_date.strftime('%Y-%m-%d')}: **{predicted_cases}**")
            
            st.markdown("---")
            st.markdown("#### Snapshot of Key Features Used for Prediction (after any imputation):")
            
            display_cols = []
            display_cols.extend([TARGET_COL_NAME] + METEO_COLS_USED) 
            display_cols.extend([f"{TARGET_COL_NAME}_lag_7"] + [f"{m}_lag_7" for m in METEO_COLS_USED]) 
            display_cols.extend(['month', 'year', 'day_of_week', 'season']) 
            if METEO_COLS_USED: # Ensure METEO_COLS_USED is not empty
                example_roll_col = f"{METEO_COLS_USED[0]}_roll_mean_7" 
                if example_roll_col in input_df.columns:
                    display_cols.append(example_roll_col)

            display_cols_unique = sorted(list(set(d for d in display_cols if d in input_df.columns)))
            if display_cols_unique:
                st.dataframe(input_df[display_cols_unique].style.format(precision=2))
            else:
                st.write("No key features available to display based on current configuration.")


        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.error("Details of the input DataFrame being processed (ensure all columns are correct):")
            st.dataframe(input_df)


else:
    st.warning("Could not load model artifacts. Please ensure the training script has been run successfully and artifacts are in the correct directory ('model_artifacts/').")

st.markdown("---")
st.markdown("App using pre-trained model.")


#**Before running:**

#1.  **Place `de.csv`:** Make sure the `de.csv` file is in the same directory where you will run the `train_dengue_model.py` script (the first script).
#2.  **Run Training First:** Execute the first script (e.g., `python train_dengue_model.py`). This will read `de.csv`, train the model, and save the artifacts in the `model_artifacts` folder.
#3.  **Run Streamlit App:** Then, run the second script (e.g., `streamlit run app_dengue_predictor.py`). It will load the artifacts and launch the web application.

#I've added more robust checks for column existence and warnings in the training script. The app now also loads the `target_col_name.pkl` to be absolutely sure about the target column's name when constructing feature keys for user input and display. The parameter grid for `GridSearchCV` in the training script is slightly reduced for faster execution during this update; you can expand it again for a more thorough hyperparameter sear
