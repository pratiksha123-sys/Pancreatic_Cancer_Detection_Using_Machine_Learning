# app.py - Flask Backend for Pancreatic Cancer Detection (Self-Contained Training)

import pandas as pd
import numpy as np
import sqlite3
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import json
import joblib # Import joblib for loading/saving models and preprocessors
from reportlab.lib.pagesizes import letter # type: ignore
from reportlab.pdfgen import canvas  # type: ignore
from flask import send_file
import io
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer # type: ignore
from reportlab.lib.styles import getSampleStyleSheet # type: ignore
from reportlab.lib import colors # type: ignore

# IMPORTANT: If you see "Import "imblearn.over_sampling" could not be resolved" error,
# run 'pip install imbalanced-learn' in your terminal.
from imblearn.over_sampling import SMOTE # type: ignore

# IMPORTANT: If you see "Import "xgboost" could not be resolved" error,
# run 'pip install xgboost' in your terminal.
import xgboost as xgb

# Import other models for comparison
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_super_secret_key_here' # Replace with a strong secret key

# --- Prediction Threshold Configuration ---
# Adjust this value to control the sensitivity of cancer detection.
# A higher value (e.g., 0.7, 0.8) makes the model more conservative (fewer false positives, but potentially more false negatives).
# A lower value (e.g., 0.3, 0.4) makes the model more aggressive (more true positives, but potentially more false positives).
CANCER_THRESHOLD = 0.5 # Neutral threshold for balancing false positives/negatives

# --- Database Setup ---
DATABASE = 'users.db'

def init_db():
    """
    Initializes the SQLite database for user storage.
    Creates 'users' table only if it doesn't exist.
    """
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        #  Use CREATE TABLE IF NOT EXISTS to preserve data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT UNIQUE,
                full_name TEXT,
                date_of_birth TEXT,
                country TEXT,
                gender TEXT,
                phone_number TEXT
            )
        ''')
        conn.commit()
    print("User table initialized (if not already present).")

def create_prediction_table():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            Country TEXT,
            Age INTEGER,
            Gender TEXT,
            Smoking_History TEXT,
            Obesity TEXT,
            Diabetes TEXT,
            Chronic_Pancreatitis TEXT,
            Family_History TEXT,
            Hereditary_Condition TEXT,
            Jaundice TEXT,
            Abdominal_Discomfort TEXT,
            Back_Pain TEXT,
            Weight_Loss TEXT,
            Development_of_Type2_Diabetes TEXT,
            Alcohol_Consumption TEXT,
            Physical_Activity_Level TEXT,
            Diet_Processed_Food TEXT,
            prediction TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("Prediction table with all 24 features created successfully.")
    
    # Add this after creating your user table
def create_login_history():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS login_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        login_time TEXT,
        logout_time TEXT
    )
''')


#  Call both functions ONCE inside app context
with app.app_context():
    init_db()
    create_prediction_table()
    create_login_history()


# --- Global Model and Preprocessing Components ---
model = None # This will store the BEST trained model
label_encoders = {}
scaler = None
feature_columns = [] # To store the names of important features used by the model
numerical_cols_trained = [] # To store the names of numerical columns the scaler was trained on
model_performances = {} # To store metrics for all trained models

MODEL_PATH = 'pancreatic_cancer_model.pkl' # Path for the BEST model
LABEL_ENCODERS_PATH = 'label_encoders.pkl'
SCALER_PATH = 'scaler.pkl'
FEATURE_COLUMNS_PATH = 'feature_columns.json'
NUMERICAL_COLS_TRAINED_PATH = 'numerical_cols_trained.json'
MODEL_PERFORMANCE_PATH = 'model_performances.json' # Path for all model metrics

def train_and_save_model_components():
    """
    Loads dataset, preprocesses, trains multiple ML models, compares them,
    saves the best model and all model performance metrics locally.
    This function is called if model files are not found.
    """
    global model, label_encoders, scaler, feature_columns, numerical_cols_trained, model_performances

    print("Model components not found or outdated. Training and saving models locally...")
    try:
        # --- 1. Load and Prepare Data ---
        # Ensure 'pancreatic_cancer_prediction_sample.csv' is in the same directory as app.py
        df = pd.read_csv('pancreatic_cancer_prediction_sample.csv')
        
        # --- Remove unwanted features ---
        remove_cols = ['Stage_at_Diagnosis', "Treatment_Type", 'Survival_Time_Months',
                    "Access_to_Healthcare", "Urban_vs_Rural", "Economic_Status"]
        df.drop(columns=remove_cols, inplace=True, errors='ignore')
        df.drop_duplicates(inplace=True)

        binary_cols_to_convert_to_object = [
            'Smoking_History', 'Obesity', 'Diabetes', 'Chronic_Pancreatitis',
            'Family_History', 'Hereditary_Condition', 'Jaundice',
            'Abdominal_Discomfort', 'Back_Pain', 'Weight_Loss',
            'Development_of_Type2_Diabetes', 'Alcohol_Consumption'
        ]
        for col in binary_cols_to_convert_to_object:
            if col in df.columns:
                df[col] = df[col].astype(str)

        X = df.drop('Survival_Status', axis=1)
        y = df['Survival_Status']

        original_numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        original_categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

        print(f"Original Categorical Columns (for Label Encoding): {original_categorical_cols}")
        print(f"Original Numerical Columns (for Scaling): {original_numerical_cols}")

        # --- 2. Preprocessing (Fit and Transform) ---
        temp_label_encoders = {}
        X_transformed = X.copy()

        for col in original_categorical_cols:
            le = LabelEncoder()
            X_transformed[col] = le.fit_transform(X_transformed[col])
            temp_label_encoders[col] = le

        temp_scaler = StandardScaler()
        if original_numerical_cols:
            X_transformed[original_numerical_cols] = temp_scaler.fit_transform(X_transformed[original_numerical_cols])

        # Assign to global variables
        label_encoders = temp_label_encoders
        scaler = temp_scaler
        numerical_cols_trained = original_numerical_cols
        feature_columns = X_transformed.columns.tolist() # All features after preprocessing

        # Apply SMOTE to training data to handle imbalance
        print("Applying SMOTE to balance training data...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_transformed, y)
        print(f"Original samples: {len(y)}")
        print(f"Resampled samples: {len(y_resampled)}")
        print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts()}")

        # Split data for evaluation (after SMOTE, if applicable)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

        # --- 3. Train and Evaluate Multiple Models ---
        trained_models = {}
        current_model_performances = {}
        best_accuracy = -1
        best_model_name = ""
        best_model_instance = None

        # XGBoost Classifier
        neg_count = y_resampled.value_counts()[0]
        pos_count = y_resampled.value_counts()[1]
        scale_pos_weight_value = neg_count / pos_count
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            n_estimators=500,
            learning_rate=0.03,
            min_child_weight=7,
            gamma=0.1,
            scale_pos_weight=scale_pos_weight_value
        )
        xgb_model.fit(X_train, y_train)
        y_pred_xgb = xgb_model.predict(X_test)
        current_model_performances["XGBoost"] = {
            "accuracy": round(accuracy_score(y_test, y_pred_xgb), 4),
            "precision": round(precision_score(y_test, y_pred_xgb), 4),
            "recall": round(recall_score(y_test, y_pred_xgb), 4),
            "f1_score": round(f1_score(y_test, y_pred_xgb), 4)
        }
        trained_models["XGBoost"] = xgb_model
        print(f"XGBoost Metrics: {current_model_performances['XGBoost']}")

        # Random Forest Classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        current_model_performances["Random Forest"] = {
            "accuracy": round(accuracy_score(y_test, y_pred_rf), 4),
            "precision": round(precision_score(y_test, y_pred_rf), 4),
            "recall": round(recall_score(y_test, y_pred_rf), 4),
            "f1_score": round(f1_score(y_test, y_pred_rf), 4)
        }
        trained_models["Random Forest"] = rf_model
        print(f"Random Forest Metrics: {current_model_performances['Random Forest']}")

        # Logistic Regression
        lr_model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        current_model_performances["Logistic Regression"] = {
            "accuracy": round(accuracy_score(y_test, y_pred_lr), 4),
            "precision": round(precision_score(y_test, y_pred_lr), 4),
            "recall": round(recall_score(y_test, y_pred_lr), 4),
            "f1_score": round(f1_score(y_test, y_pred_lr), 4)
        }
        trained_models["Logistic Regression"] = lr_model
        print(f"Logistic Regression Metrics: {current_model_performances['Logistic Regression']}")

        # Determine the best model based on accuracy
        for name, metrics in current_model_performances.items():
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_model_name = name
                best_model_instance = trained_models[name]

        model = best_model_instance # Assign the best model to the global 'model' variable
        model_performances = current_model_performances # Assign all performances

        print(f"Best model selected: {best_model_name} with Accuracy: {best_accuracy}")

        # --- 4. Save Components ---
        joblib.dump(model, MODEL_PATH) # Save only the best model
        joblib.dump(label_encoders, LABEL_ENCODERS_PATH)
        joblib.dump(scaler, SCALER_PATH)
        with open(FEATURE_COLUMNS_PATH, 'w') as f:
            json.dump(feature_columns, f)
        with open(NUMERICAL_COLS_TRAINED_PATH, 'w') as f:
            json.dump(numerical_cols_trained, f)
        with open(MODEL_PERFORMANCE_PATH, 'w') as f:
            json.dump(model_performances, f, indent=4) # Save all model performances

        print("All model components trained and saved successfully locally.")

    except Exception as e:
        print(f"ERROR: Failed to train and save model components locally: {e}")
        print("Please ensure 'pancreatic_cancer_prediction_sample.csv' is in the same directory.")
        exit(1) # Exit if training/saving fails

def load_model_components():
    """
    Loads a pre-trained model, label encoders, scaler, and feature lists from files.
    If files are not found, it triggers training.
    """
    global model, label_encoders, scaler, feature_columns, numerical_cols_trained, model_performances

    # Check if all necessary files exist
    required_files = [
        MODEL_PATH, LABEL_ENCODERS_PATH, SCALER_PATH,
        FEATURE_COLUMNS_PATH, NUMERICAL_COLS_TRAINED_PATH, MODEL_PERFORMANCE_PATH
    ]
    if not all(os.path.exists(f) for f in required_files):
        train_and_save_model_components() # If any file is missing, train them all

    try:
        model = joblib.load(MODEL_PATH)
        label_encoders = joblib.load(LABEL_ENCODERS_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(FEATURE_COLUMNS_PATH, 'r') as f:
            feature_columns = json.load(f)
        with open(NUMERICAL_COLS_TRAINED_PATH, 'r') as f:
            numerical_cols_trained = json.load(f)
        with open(MODEL_PERFORMANCE_PATH, 'r') as f:
            model_performances = json.load(f)

        print("Model, encoders, scaler, feature columns, and model performances loaded successfully from files.")
        print(f"Features expected by the model: {feature_columns}")
        print(f"Numerical columns for scaling: {numerical_cols_trained}")
        print(f"Loaded Model Performances: {model_performances}")
    except Exception as e:
        print(f"ERROR: Failed to load model components after training/initial check: {e}")
        print("Please ensure the .pkl and .json files are not corrupted and are compatible with the scikit-learn/xgboost version installed.")
        exit(1) # Exit if loading fails


# Load the model components when the app starts
with app.app_context():
    load_model_components()

# --- Utility Functions ---
def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row # This allows accessing columns by name
    return conn

# --- Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    """Handles user registration with extended details."""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    full_name = data.get('full_name')
    date_of_birth = data.get('date_of_birth')
    country = data.get('country')
    gender = data.get('gender')
    phone_number = data.get('phone_number')

    if not username or not password:
        return jsonify({'message': 'Username and password are required'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password, email, full_name, date_of_birth, country, gender, phone_number) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (username, password, email, full_name, date_of_birth, country, gender, phone_number)
        )
        conn.commit()
        return jsonify({'message': 'User registered successfully'}), 201
    except sqlite3.IntegrityError as e:
        if "UNIQUE constraint failed: users.username" in str(e):
            return jsonify({'message': 'Username already exists'}), 409
        elif "UNIQUE constraint failed: users.email" in str(e):
            return jsonify({'message': 'Email already exists'}), 409
        else:
            return jsonify({'message': f'Database error: {e}'}), 500
    except Exception as e:
        print(f"Error during registration: {e}")
        return jsonify({'message': 'Internal server error'}), 500
    finally:
        conn.close()

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    identifier = data.get('identifier')
    password = data.get('password')

    if not identifier or not password:
        return jsonify({'message': 'Username/Email and password are required'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE (username = ? OR email = ?) AND password = ?",
        (identifier, identifier, password)
    )
    user = cursor.fetchone()

    if user:
        session['username'] = user['username']  # Store session

        #  Insert login history here
        from datetime import datetime
        cursor.execute(
            "INSERT INTO login_history (username, login_time) VALUES (?, ?)",
            (user['username'], datetime.now())
        )
        conn.commit()  # Commit the insert

        conn.close()
        return jsonify({'message': 'Login successful', 'username': user['username']}), 200
    else:
        conn.close()
        return jsonify({'message': 'Invalid username/email or password'}), 401


@app.route('/detect', methods=['POST'])
def detect_cancer():
    """
    Handles cancer detection requests.
    Expects a JSON payload with features.
    Uses the best trained model for prediction.
    """
    if model is None:
        return jsonify({'message': 'Model not loaded. Please check server logs.'}), 503

    data = request.get_json()
    print(f"Received data from frontend: {data}") # Debugging print

    if not data:
        return jsonify({'message': 'No input data provided'}), 400

    # Create a DataFrame from the input data
    input_df = pd.DataFrame([data])
    print(f"Initial input_df: \n{input_df}") # Debugging print

    # Ensure all *expected* important features are present in the input_df
    # Fill missing ones with a default (e.g., 0 or mode from training)
    for col in feature_columns:
        if col not in input_df.columns:
            # For categorical, fill with an empty string or the most frequent category from training
            # For numerical, fill with 0 or mean/median from training
            if col in label_encoders: # It's a categorical column if it has a label encoder
                input_df[col] = '' # Placeholder for missing categorical
            else:
                input_df[col] = 0 # Default for missing numerical

    # Keep only the selected important features and ensure their order
    input_df = input_df[feature_columns]
    print(f"Input_df after feature selection and reordering: \n{input_df}") # Debugging print


    # Apply preprocessing steps used during training *only to the relevant selected features*
    processed_input = input_df.copy() # Create a copy to avoid SettingWithCopyWarning

    # Apply Label Encoding to categorical columns among the selected features
    for col in label_encoders.keys(): # Iterate only through columns that *had* label encoders
        if col in processed_input.columns: # Check if this categorical column is in the current input
            encoder = label_encoders[col]
            # Ensure the value is treated as a string for label encoding
            val = str(processed_input[col].iloc[0])
            
            # Handle unseen categories during prediction
            if val in encoder.classes_:
                processed_input[col] = encoder.transform([val])[0]
            else:
                # If an unseen category, assign a default value (e.g., -1 or 0, or the most frequent class)
                # For simplicity, assigning -1. Ensure your model can handle this.
                # A more robust solution might map to the most frequent class or a specific 'unknown' class.
                processed_input[col] = -1
            print(f"Label encoded '{col}': original '{val}' -> encoded '{processed_input[col].iloc[0]}'")
        else:
            print(f"Warning: Categorical column '{col}' expected but not found in processed_input for label encoding.")


    # Scale numerical features among the selected features
    if scaler and numerical_cols_trained:
        # Filter processed_input to only include numerical columns that the scaler was trained on
        numerical_features_in_input = [col for col in numerical_cols_trained if col in processed_input.columns]
        
        if numerical_features_in_input:
            # Ensure the data type is numeric before scaling
            for col in numerical_features_in_input:
                processed_input[col] = pd.to_numeric(processed_input[col], errors='coerce') # Convert to numeric, coerce errors to NaN
            processed_input.fillna(0, inplace=True) # Fill any NaNs created by coercion with 0 or a sensible default

            processed_input[numerical_features_in_input] = scaler.transform(processed_input[numerical_features_in_input])
            print(f"Scaled numerical features: {numerical_features_in_input}")
        else:
            print("No numerical features found in input for scaling.")
    print(f"Input_df after all preprocessing: \n{processed_input}")


    try:
        # Get probabilities for both classes (0: No Cancer, 1: Cancer)
        prediction_proba = model.predict_proba(processed_input)[0]
        probability_no_cancer = float(prediction_proba[0]) # Explicitly cast to float
        probability_cancer = float(prediction_proba[1]) # Explicitly cast to float

        # Apply the custom threshold for final prediction message
        if probability_cancer >= CANCER_THRESHOLD:
            prediction_label = 1
            message = 'Cancer detected'
        else:
            prediction_label = 0
            message = 'No cancer detected'

        result = {
            'prediction': prediction_label,
            'probability_no_cancer': round(probability_no_cancer, 4),
            'probability_cancer': round(probability_cancer, 4),
            'message': message
        }
        # Store prediction result in prediction_history table
        try:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            username = session.get('username', 'guest')  # fallback if not logged in
            input_data_str = str(data)
            prediction_str = str(prediction_label)

            cursor.execute('''
                INSERT INTO prediction_history (
                    username, Country, Age, Gender, Smoking_History, Obesity,
                    Diabetes, Chronic_Pancreatitis, Family_History, Hereditary_Condition,
                    Jaundice, Abdominal_Discomfort, Back_Pain, Weight_Loss,
                    Development_of_Type2_Diabetes,Alcohol_Consumption, Physical_Activity_Level,
                    Diet_Processed_Food, prediction
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                username,
                data.get('Country'),
                data.get('Age'),
                data.get('Gender'),
                data.get('Smoking_History'),
                data.get('Obesity'),
                data.get('Diabetes'),
                data.get('Chronic_Pancreatitis'),
                data.get('Family_History'),
                data.get('Hereditary_Condition'),
                data.get('Jaundice'),
                data.get('Abdominal_Discomfort'),
                data.get('Back_Pain'),
                data.get('Weight_Loss'),
                data.get('Development_of_Type2_Diabetes'),
                data.get('Alcohol_Consumption'),
                data.get('Physical_Activity_Level'),
                data.get('Diet_Processed_Food'),
                str(prediction_label)
            ))

            conn.commit()
            conn.close()
            print("Prediction record stored successfully.")
        except Exception as db_error:
            print(f"Error saving prediction to DB: {db_error}")

        # Show result on console and send to frontend
        print(f"Prediction result: {result}")
        return jsonify(result), 200
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'message': f'Prediction error: {str(e)}'}), 500
    
    
@app.route('/download_result', methods=['POST'])
def download_result():
    data = request.get_json()
    if not data:
        return jsonify({'message': 'No data received for PDF generation'}), 400

    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            topMargin=30,
            bottomMargin=30,
            leftMargin=40,
            rightMargin=40
        )
        styles = getSampleStyleSheet()
        elements = []

        # Title (small spacing to save space)
        title_style = styles['Title']
        title_style.fontSize = 14
        title_style.leading = 16
        elements.append(Paragraph("Pancreatic Cancer Detection Report", title_style))
        elements.append(Spacer(1, 6))

        # Prepare table content
        table_data = [["Field", "Value"]]  # Table header
        for key, value in data.items():
            label = key.replace("_", " ").title()
            table_data.append([label, str(value)])

        # Create and format the table
        table = Table(table_data, colWidths=[180, 340])  # Adjusted widths to fit on one page
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2E86C1")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),  # Smaller font
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),  # Tighter spacing
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ]))

        elements.append(table)
        doc.build(elements)
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name="detection_result.pdf",
            mimetype='application/pdf'
        )
    except Exception as e:
        print("PDF generation error:", e)
        return jsonify({'message': 'Failed to generate PDF'}), 500


@app.route('/analysis', methods=['GET'])
def get_analysis():
    """
    Provides detailed analysis data for the frontend to render charts.
    """
    try:
        df = pd.read_csv('pancreatic_cancer_prediction_sample.csv')
        
        # Overall Survival Status
        survival_status_counts = df['Survival_Status'].value_counts().to_dict()
        
        # Gender Distribution
        gender_counts = df['Gender'].value_counts().to_dict()

        # Smoking History Distribution
        smoking_history_counts = df['Smoking_History'].value_counts().to_dict()
        
        # Age Distribution (simple bins for demonstration)
        age_bins = [0, 40, 50, 60, 70, 80, 90, 100, 120]
        age_labels = ['0-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99', '100+']
        df['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)
        age_group_counts = df['Age_Group'].value_counts().sort_index().to_dict()

        # Cancer Cases by Gender
        cancer_by_gender = df[df['Survival_Status'] == 1]['Gender'].value_counts().to_dict()
        
        analysis_data = {
            'total_records': len(df),
            'survival_status_counts': survival_status_counts, # {0: count_no_cancer, 1: count_cancer}
            'gender_distribution': gender_counts,
            'smoking_history_distribution': smoking_history_counts,
            'age_group_distribution': age_group_counts,
            'cancer_cases_by_gender': cancer_by_gender
        }
        return jsonify(analysis_data), 200
    except Exception as e:
        print(f"Error during analysis: {e}")
        return jsonify({'message': f'Error retrieving analysis data: {str(e)}'}), 500


@app.route('/model_comparison', methods=['GET'])
def get_model_comparison():
    """
    Returns the performance metrics of all trained models.
    """
    global model_performances
    if not model_performances:
        # If for some reason performances are not loaded, try to load/train
        load_model_components() 
    
    if model_performances:
        return jsonify(model_performances), 200
    else:
        return jsonify({'message': 'Model comparison data not available. Please ensure models are trained.'}), 500


@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/logout', methods=['POST'])
def logout():
    if 'username' in session:
        username = session['username']
        try:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()

            # Get the most recent login with NULL logout_time
            cursor.execute('''
                SELECT id FROM login_history 
                WHERE username = ? AND logout_time IS NULL 
                ORDER BY login_time DESC 
                LIMIT 1
            ''', (username,))
            row = cursor.fetchone()

            if row:
                login_id = row[0] if not isinstance(row, sqlite3.Row) else row['id']
                # Use Python's datetime for same format as login_time
                logout_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                cursor.execute('''
                    UPDATE login_history
                    SET logout_time = ?
                    WHERE id = ?
                ''', (logout_time, login_id))
                conn.commit()

            # Delete user account
            cursor.execute('DELETE FROM users WHERE username = ?', (username,))
            conn.commit()

            conn.close()
        except Exception as e:
            print("Error during logout:", e)

    session.clear()
    return jsonify({'message': 'Logged out successfully'}), 200



if __name__ == '__main__':
    # Load the model components when the script is run directly
    load_model_components()
    # Run Flask app
    app.run(debug=True, port=5000) # Set debug=False in production
