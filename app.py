# Import the libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    matthews_corrcoef,
    confusion_matrix, 
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============= GITHUB TEST DATA LINK  =============
# Change this URL to your GitHub raw test data link
GITHUB_TEST_DATA_URL = "https://raw.githubusercontent.com/kamaraj-bitspilani/2025AA05921/main/data/test_data.csv"

# Page configuration 
st.set_page_config(
    page_title="ML Assignment 2 - Loan Approval Classification",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS  single-page layout
st.markdown("""
<style>
body { margin: 0; padding: 0; }
.main { padding: 0.5rem; }
.metric-value { font-size: 1.8rem; font-weight: bold; }
.metric-label { font-size: 0.75rem; opacity: 0.8; }
[data-testid="stVerticalBlock"] { gap: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# function to download the data from the github repository
def download_test_data_from_github():
    """Download test data directly from Personal GitHub repository."""
    try:
        response = requests.get(GITHUB_TEST_DATA_URL)
        if response.status_code == 200: # Positive response
            return pd.read_csv(BytesIO(response.content))
        else:
            st.error(f"❌ Failed to download from GitHub: HTTP {response.status_code}")
            return None
    except Exception as e:
        st.error(f"❌ Error downloading test data from GitHub: {str(e)}")
        return None


# cached version of the download function to avoid repeated downloads during app interactions
@st.cache_data
def cached_download_test_data():
    """Cached version of downloading test data from GitHub."""
    return download_test_data_from_github()

# save the test data to the system 
def save_test_data_to_system(df):
    """Prepare test data for download to professor's system."""
    try:
        # Convert DataFrame to CSV format
        csv_data = df.to_csv(index=False)
        return csv_data
    except Exception as e:
        st.error(f"❌ Error preparing test data: {str(e)}")
        return None


# Load all models, encoders, and scalers - cached to optimize performance
@st.cache_resource
def load_models():
    """Load all trained models from the model folder (cached)."""
    model_files = {
        "1.Logistic Regression": {
            "model": "model/1_logistic_regression_model.joblib",
            "encoders": "model/1_logistic_regression_encoders.joblib",
            "scaler": "model/1_logistic_regression_scaler.joblib"
        },
        "2.Decision Tree": {
            "model": "model/2_decision_tree_model.joblib",
            "encoders": "model/2_decision_tree_encoders.joblib",
            "scaler": None
        },
        "3.K-Nearest Neighbors": {
            "model": "model/3_knn_model.joblib",
            "encoders": "model/3_knn_encoders.joblib",
            "scaler": "model/3_knn_scaler.joblib"
        },
        "4.Naive Bayes": {
            "model": "model/4_naive_bayes_model.joblib",
            "encoders": "model/4_naive_bayes_encoders.joblib",
            "scaler": None
        },
        "5.Random Forest": {
            "model": "model/5_random_forest_model.joblib",
            "encoders": "model/5_random_forest_encoders.joblib",
            "scaler": None
        },
        "6.XGBoost": {
            "model": "model/6_xgboost_model.joblib",
            "encoders": "model/6_xgboost_encoders.joblib",
            "scaler": None
        }
    }
    
    models = {}
    encoders_dict = {}
    scalers_dict = {}
    failed_models = []
    
    for name, paths in model_files.items():
        try:
            # Load model
            if os.path.exists(paths["model"]):
                models[name] = joblib.load(paths["model"])
            else:
                failed_models.append(name)
                continue
            
            # Load encoders
            if paths["encoders"] and os.path.exists(paths["encoders"]):
                encoders_dict[name] = joblib.load(paths["encoders"])
            
            # Load scaler if it exists
            if paths["scaler"] and os.path.exists(paths["scaler"]):
                scalers_dict[name] = joblib.load(paths["scaler"])
                
        except Exception as e:
            failed_models.append(name)
    
    return models, encoders_dict, scalers_dict, len(failed_models) == 0

# Function to load test data with optional user upload, GitHub download, or default file, and apply encoders/scalers
def load_test_data(uploaded_file=None, selected_model_name=None, encoders_dict=None, scalers_dict=None, use_github=False):
    """
    Load test data from user upload, GitHub, or default test_data.csv.
    Apply saved encoders and scalers based on selected model.
    
    Args:
        uploaded_file: Streamlit uploaded file object (optional)
        selected_model_name: Name of selected model
        encoders_dict: Dictionary of saved encoders
        scalers_dict: Dictionary of saved scalers
        use_github: Boolean to download from GitHub
    
    Returns:
        X_test, y_test: Feature matrix and target vector
    """
    try:
        if uploaded_file is not None:
            # User uploaded a file
            df = pd.read_csv(uploaded_file)
        elif use_github:
            # Download from GitHub
            df = download_test_data_from_github()
            if df is None:
                return None, None
        else:
            # Use default test data
            df = pd.read_csv('data/test_data.csv')
        
        # Check if data loaded properly
        if df.empty:
            st.error("❌ Loaded dataset is empty")
            return None, None
        
        # Preprocess: Encode categorical features using saved encoders
        categorical_cols = [
            'person_gender',
            'person_education',
            'person_home_ownership',
            'loan_intent',
            'previous_loan_defaults_on_file'
        ]
        
        df_processed = df.copy()
        
        # Use saved encoders if available for the selected model
        if selected_model_name and selected_model_name in encoders_dict:
            label_encoders = encoders_dict[selected_model_name]
            for col in categorical_cols:
                if col in df_processed.columns and col in label_encoders:
                    try:
                        df_processed[col] = label_encoders[col].transform(df_processed[col].astype(str))
                    except Exception as e:
                        # If encoding fails, use LabelEncoder for this column
                        le = LabelEncoder()
                        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        else:
            # Fallback: Create new encoders if saved ones not available
            for col in categorical_cols:
                if col in df_processed.columns:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        
        # Separate features and target
        if 'loan_status' in df_processed.columns:
            X_test = df_processed.drop('loan_status', axis=1)
            y_test = df_processed['loan_status']
        else:
            # If loan_status not found, assume last column is target
            X_test = df_processed.iloc[:, :-1]
            y_test = df_processed.iloc[:, -1]
        
        # Apply scaler if available for the selected model
        if selected_model_name and selected_model_name in scalers_dict:
            scaler = scalers_dict[selected_model_name]
            X_test = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns
            )
        
        return X_test, y_test
        
    except FileNotFoundError:
        st.error("❌ Default test data file not found at data/test_data.csv")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading test data: {str(e)}")
        return None, None

# Function to evaluate a single model on test data and return all metrics
def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a single model on test data.
    
    Args:
        model: Trained ML model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model
    
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get probability predictions if available
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)
            # For binary classification, use positive class probability
            if y_prob.shape[1] == 2:
                y_prob_positive = y_prob[:, 1]
            else:
                y_prob_positive = None
        else:
            y_prob_positive = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # AUC Score (only for binary classification with probabilities)
        try:
            if y_prob_positive is not None and len(np.unique(y_test)) == 2:
                auc = roc_auc_score(y_test, y_prob_positive)
            else:
                auc = None
        except:
            auc = None
        
        # Other metrics
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_pred': y_pred,
            'y_prob': y_prob_positive
        }
        
    except Exception as e:
        st.error(f"❌ Error evaluating {model_name}: {str(e)}")
        return None

# Function to display metrics in a clean table format
def display_metrics_table(results):
    """Display evaluation metrics in a clean table format."""
    metrics_data = {
        'Metric': ['Accuracy', 'AUC Score', 'Precision', 'Recall', 'F1 Score'],
        'Value': [
            f"{results['accuracy']:.4f}",
            f"{results['auc']:.4f}" if results['auc'] is not None else "N/A",
            f"{results['precision']:.4f}",
            f"{results['recall']:.4f}",
            f"{results['f1']:.4f}"
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df)

# Function to display confusion matrix 
def display_confusion_matrix(cm):
    """Display confusion matrix as a compact heatmap."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True, annot_kws={'size': 10})
    ax.set_title("Confusion Matrix", fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)
    st.pyplot(fig, use_container_width=True)
    plt.close()

def main():
    """Main application function with fixed left panel and metrics table."""
    
    # Load models once (cached)
    models, encoders_dict, scalers_dict, models_loaded_ok = load_models()
    
    if not models:
        st.error("❌ No models loaded. Please ensure .joblib files exist in the 'model' folder.")
        st.stop()
    
    # Main Layout: 3 Columns (30% left instructions, 30% middle controls, 40% right results)
    left_col, middle_col, right_col = st.columns([0.30, 0.30, 0.40], gap="medium")
    
    # ============== LEFT COLUMN (30%) - INSTRUCTIONS ONLY ==============
    with left_col:
        # Try to display the BITS Pilani logo if present in the workspace
        logo_paths = [
            "BITS_Pilani-Logo.svg.png",
            "assets/BITS_Pilani-Logo.svg.png",
            "data/BITS_Pilani-Logo.svg.png",
            "model/BITS_Pilani-Logo.svg.png"
        ]
        logo_shown = False
        for lp in logo_paths:
            if os.path.exists(lp):
                try:
                    st.image(lp, width=140)
                    logo_shown = True
                    break
                except Exception:
                    # ignore any image loading errors and try next path
                    pass

        # Display BITS ID and Name below logo (always shown)
        st.markdown("""**BITS-ID:** 2025AA05921  
**Name:** Kamaraj""")
        
        st.markdown('<h3 style="margin-top: 1rem;">📖 Instructions</h3>', unsafe_allow_html=True)
        
        # Instructions Section (Fixed, no minimize)
        st.markdown("**How to Use:**")
        st.markdown("""
1. **📥 Download Test Data** - Get  the test data CSV file
2. **📁 Upload Test Data** - Upload the test data downloaded or use default test data already in the app
3. **🎯 Select Model** - Choose ML model
4. **🚀 Run Evaluation** - Click  it
5. **📊 View Results** - See metrics in the Results
        """)
        
        # Model load status
        if models_loaded_ok and len(models) == 6:
            st.success(f"{len(models)} Models Loaded", icon="✅")
        else:
            st.warning(f"⚠️ {len(models)}/6 Models")
    
    # ============== MIDDLE COLUMN (30%) - CONTROLS ==============
    with middle_col:
        # Title
        st.markdown('<h3 style="margin-top: 0;">Machine Learning Assignment-2<br/>Loan Approval Classification</h3>', unsafe_allow_html=True)
        
        # Single Button - Download TestData
        st.markdown("**1. 📥 Download**")
        with st.spinner("Preparing..."):
            github_data = cached_download_test_data()
            if github_data is not None:
                csv_data = save_test_data_to_system(github_data)
                if csv_data is not None:
                    st.download_button(
                        label="📥 Download TestData.csv",
                        data=csv_data,
                        file_name="test_data.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_testdata_btn"
                    )
        
        # File uploader
        st.markdown("**2. 📁 Upload Test Data**")
        uploaded_file = st.file_uploader(
            "CSV file",
            type=["csv"],
            label_visibility="collapsed",
            help="Upload CSV"
        )
        
        # Model selection
        st.markdown("**3. 🎯 Select Model**")
        selected_model_name = st.selectbox(
            "Model",
            list(models.keys()),
            label_visibility="collapsed",
            key="model_select"
        )
        
        # Load test data with selected model's encoders and scalers (only when needed)
        X_test, y_test = load_test_data(
            uploaded_file=uploaded_file,
            selected_model_name=selected_model_name,
            encoders_dict=encoders_dict,
            scalers_dict=scalers_dict,
            use_github=False
        )
        
        if X_test is None or y_test is None:
            st.error("❌ Unable to load data")
            st.stop()
        
        # Dataset info message
        if uploaded_file is not None:
            st.info(f"File: {uploaded_file.name[:20]}", icon="📊")
        else:
            st.info("Data: Default Test Data", icon="📊")
        
        # Evaluation button
        st.markdown("**4. Run Evaluation**")
        if st.button("4. 🚀Run Evaluation", type="primary", use_container_width=True, key="eval_btn"):
            with st.spinner(f"Evaluating..."):
                selected_model = models[selected_model_name]
                results = evaluate_model(selected_model, X_test, y_test, selected_model_name)
                
                if results is not None:
                    st.session_state['results'] = results
                    st.session_state['selected_model_name'] = selected_model_name
                    st.success(f"✅ {selected_model_name} executed!")
    
    # ============== RIGHT COLUMN (40%) - RESULTS ==============
    with right_col:
        if 'results' in st.session_state:
            model_name = st.session_state.get('selected_model_name', 'Model')
            st.markdown(f'<h3 style="margin-top: 0;">5. 📊 Results-> Model-  {model_name}</h3>', unsafe_allow_html=True)
            
            results = st.session_state['results']
            
            # Metrics table
            st.markdown("**📈 Evaluation Metrics**")
            display_metrics_table(results)
            
            # Confusion Matrix
            st.markdown("**🎯 Confusion Matrix**")
            display_confusion_matrix(results['confusion_matrix'])
        else:
            st.markdown('<h3 style="margin-top: 0;">📊 Results</h3>', unsafe_allow_html=True)
            st.info("Results appear here", icon="ℹ️")


if __name__ == "__main__":
    main() # main function start