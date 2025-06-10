import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import torch with error handling
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError as e:
    st.error(f"PyTorch import error: {e}")
    TORCH_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="LoanScopeAI - Loan Eligibility Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .prediction-success {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #c3e6cb;
    }
    .prediction-danger {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #f5c6cb;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d5aa7;
    }
</style>
""", unsafe_allow_html=True)

# Model Classes (only define if torch is available)
if TORCH_AVAILABLE:
    class ResidualBlock(nn.Module):
        """Residual block for MLP with skip connection."""
        def __init__(self, in_dim, out_dim, dropout=0.2):
            super().__init__()
            self.linear1 = nn.Linear(in_dim, out_dim)
            self.linear2 = nn.Linear(out_dim, out_dim)
            self.bn1 = nn.BatchNorm1d(out_dim)
            self.bn2 = nn.BatchNorm1d(out_dim)
            self.dropout = nn.Dropout(dropout)
            self.activation = nn.GELU()
            
            # Skip connection
            self.shortcut = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        
        def forward(self, x):
            identity = self.shortcut(x)
            
            out = self.linear1(x)
            out = self.bn1(out)
            out = self.activation(out)
            out = self.dropout(out)
            
            out = self.linear2(out)
            out = self.bn2(out)
            
            out += identity
            out = self.activation(out)
            out = self.dropout(out)
            return out

    class TabTransformer(nn.Module):
        """TabTransformer model architecture for tabular data."""
        def __init__(self,
                     cat_cardinalities: dict,
                     num_cont_features: int,
                     num_classes: int,
                     dim: int = 64,
                     depth: int = 8,
                     heads: int = 8,
                     mlp_dropout: float = 0.2):
            super().__init__()
            self.cat_cardinalities = cat_cardinalities
            self.num_cont_features = num_cont_features
            self.num_classes = num_classes
            
            # Embedding layers for categorical features
            self.embeddings = nn.ModuleList([
                nn.Embedding(cardinality, dim)
                for cardinality in cat_cardinalities.values()
            ])
            
            # Transformer encoder
            self.transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=heads,
                    dim_feedforward=dim * 4,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=depth
            )
            
            # Continuous features processing
            if num_cont_features > 0:
                self.cont_norm = nn.BatchNorm1d(num_cont_features)
                self.cont_proj = nn.Linear(num_cont_features, dim)
            
            # MLP head
            transformer_output_dim = len(cat_cardinalities) * dim
            mlp_input_dim = transformer_output_dim + (dim if num_cont_features > 0 else 0)
            
            self.mlp = nn.Sequential(
                nn.BatchNorm1d(mlp_input_dim),
                nn.Linear(mlp_input_dim, mlp_input_dim * 8),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.BatchNorm1d(mlp_input_dim * 8),
                
                nn.Linear(mlp_input_dim * 8, mlp_input_dim * 4),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.BatchNorm1d(mlp_input_dim * 4),
                
                ResidualBlock(mlp_input_dim * 4, mlp_input_dim * 4, dropout=mlp_dropout),
                
                nn.Linear(mlp_input_dim * 4, mlp_input_dim * 2),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.BatchNorm1d(mlp_input_dim * 2),
                
                nn.Linear(mlp_input_dim * 2, num_classes)
            )
        
        def forward(self, cat_data, cont_data):
            # Process categorical features
            if cat_data.size(1) > 0:
                embeds = [emb(cat_data[:, i]) for i, emb in enumerate(self.embeddings)]
                embeds = torch.stack(embeds, dim=1)
                transformed = self.transformer(embeds)
                transformed = transformed.flatten(1)
            else:
                transformed = torch.empty(cat_data.size(0), 0, device=cat_data.device)
            
            # Process continuous features
            if self.num_cont_features > 0 and cont_data.size(1) > 0:
                cont_data = self.cont_norm(cont_data)
                cont_proj = self.cont_proj(cont_data)
                combined = torch.cat([transformed, cont_proj], dim=1)
            else:
                combined = transformed
            
            return self.mlp(combined)

class FeatureEngineering:
    """Feature engineering pipeline for preprocessing and feature selection."""
    def __init__(self,
                 cat_cols: list,
                 cont_cols: list,
                 target_col: str,
                 k_features: int = 10):
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.target_col = target_col
        self.k_features = k_features
        self.selector = None
        self.selected_cont_cols = None
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def clean_numeric_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove special characters from numeric columns and convert to float."""
        df_clean = df.copy()
        for col in self.cont_cols:
            if col in df_clean.columns:
                df_clean[col] = (
                    df_clean[col]
                    .astype(str)
                    .str.replace(r'[$,%]', '', regex=True)
                    .replace('nan', np.nan)
                )
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        return df_clean
    
    def encode_categoricals(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Label encode categorical features."""
        df_encoded = df.copy()
        for col in self.cat_cols:
            if col in df_encoded.columns:
                df_encoded[col] = df_encoded[col].astype(str)
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
                else:
                    known_categories = set(self.label_encoders[col].classes_)
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: x if x in known_categories else self.label_encoders[col].classes_[0]
                    )
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        return df_encoded
    
    def transform_single(self, input_data: dict) -> tuple:
        """Transform single prediction input."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted")
        
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        df = self.clean_numeric_cols(df)
        df_encoded = self.encode_categoricals(df, fit=False)
        
        # Extract features
        X_cat = df_encoded[self.cat_cols].values.astype(np.int64)
        X_cont = self.scaler.transform(df_encoded[self.selected_cont_cols].values).astype(np.float32)
        
        return X_cat, X_cont

# Cache functions for loading models
@st.cache_resource
def load_model_and_pipeline():
    """Load the trained model and feature pipeline."""
    if not TORCH_AVAILABLE:
        st.error("PyTorch is not available. Cannot load the model.")
        return None, None
    
    try:
        # Check if model files exist
        pipeline_path = 'configs/feature_pipeline/feature_pipeline.pkl'
        model_path = 'models/loan_model.pth'
        
        if not os.path.exists(pipeline_path):
            st.error(f"Feature pipeline file not found: {pipeline_path}")
            return None, None
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None, None
        
        # Load feature pipeline
        with open(pipeline_path, 'rb') as f:
            pipeline_data = pickle.load(f)
        
        # Create feature engineering instance
        feat_engineer = FeatureEngineering([], [], '')
        feat_engineer.__dict__.update(pipeline_data)
        
        # Load model
        cat_cardinalities = {
            feat: len(feat_engineer.label_encoders[feat].classes_)
            for feat in feat_engineer.cat_cols
        }
        
        model = TabTransformer(
            cat_cardinalities=cat_cardinalities,
            num_cont_features=len(feat_engineer.selected_cont_cols),
            num_classes=len(feat_engineer.target_encoder.classes_),
            dim=64,
            depth=8,
            mlp_dropout=0.3
        )
        
        # Load trained weights with map_location for CPU
        device = torch.device('cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        return model, feat_engineer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure the model files are properly trained and saved.")
        return None, None

def predict_loan_status(model, feat_engineer, input_data):
    """Make prediction for loan status."""
    if not TORCH_AVAILABLE:
        st.error("PyTorch is not available for predictions.")
        return None, None, None
    
    try:
        # Transform input data
        X_cat, X_cont = feat_engineer.transform_single(input_data)
        
        # Convert to tensors
        cat_tensor = torch.tensor(X_cat, dtype=torch.long)
        cont_tensor = torch.tensor(X_cont, dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(cat_tensor, cont_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Decode prediction
        prediction = feat_engineer.target_encoder.inverse_transform([predicted_class])[0]
        
        return prediction, confidence, probabilities[0].numpy()
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

def create_gauge_chart(confidence, prediction):
    """Create a gauge chart for confidence score."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence Score<br>{prediction}"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_feature_importance_chart(feat_engineer, input_data):
    """Create a feature importance visualization."""
    # This is a simplified version - in practice, you'd want to compute actual feature importance
    features = feat_engineer.selected_cont_cols
    values = [input_data.get(feat, 0) for feat in features]
    
    fig = px.bar(
        x=features,
        y=values,
        title="Input Feature Values",
        labels={'x': 'Features', 'y': 'Values'}
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🏦 LoanScopeAI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Intelligent Loan Eligibility Prediction System</p>', unsafe_allow_html=True)
    
    # Check PyTorch availability
    if not TORCH_AVAILABLE:
        st.error("⚠️ PyTorch is not properly installed or available. Please install PyTorch to use this application.")
        st.info("To install PyTorch, run: `pip install torch torchvision torchaudio`")
        return
    
    # Load model and pipeline
    model, feat_engineer = load_model_and_pipeline()
    
    if model is None or feat_engineer is None:
        st.error("Failed to load model. Please check if model files exist in the correct directories.")
        st.info("""
        Expected file structure:
        ```
        ├── configs/
        │   └── feature_pipeline/
        │       └── feature_pipeline.pkl
        ├── models/
        │   └── loan_model.pth
        └── app.py
        ```
        """)
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Loan Prediction", "About", "Model Info"])
    
    if page == "Loan Prediction":
        loan_prediction_page(model, feat_engineer)
    elif page == "About":
        about_page()
    elif page == "Model Info":
        model_info_page(feat_engineer)

def loan_prediction_page(model, feat_engineer):
    st.markdown('<h2 class="sub-header">📊 Loan Application Form</h2>', unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        
        # Categorical features
        term = st.selectbox(
            "Loan Term",
            options=feat_engineer.label_encoders['Term'].classes_,
            help="Select the loan term period"
        )
        
        years_in_job = st.selectbox(
            "Years in Current Job",
            options=feat_engineer.label_encoders['Years in current job'].classes_,
            help="How long have you been in your current job?"
        )
        
        home_ownership = st.selectbox(
            "Home Ownership",
            options=feat_engineer.label_encoders['Home Ownership'].classes_,
            help="Your current home ownership status"
        )
        
        purpose = st.selectbox(
            "Loan Purpose",
            options=feat_engineer.label_encoders['Purpose'].classes_,
            help="What is the purpose of this loan?"
        )
        
        # Financial Information
        st.subheader("Financial Information")
        
        current_loan_amount = st.number_input(
            "Current Loan Amount ($)",
            min_value=0,
            max_value=1000000,
            value=10000,
            step=1000,
            help="Enter the requested loan amount"
        )
        
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=650,
            step=10,
            help="Your current credit score"
        )
        
        annual_income = st.number_input(
            "Annual Income ($)",
            min_value=0,
            max_value=1000000,
            value=50000,
            step=1000,
            help="Your total annual income"
        )
    
    with col2:
        st.subheader("Credit History")
        
        monthly_debt = st.number_input(
            "Monthly Debt ($)",
            min_value=0,
            max_value=50000,
            value=1000,
            step=100,
            help="Your total monthly debt payments"
        )
        
        years_credit_history = st.number_input(
            "Years of Credit History",
            min_value=0.0,
            max_value=50.0,
            value=10.0,
            step=0.5,
            help="How many years of credit history do you have?"
        )
        
        months_since_delinquent = st.number_input(
            "Months Since Last Delinquent",
            min_value=0,
            max_value=200,
            value=12,
            step=1,
            help="Months since your last delinquent payment"
        )
        
        num_open_accounts = st.number_input(
            "Number of Open Accounts",
            min_value=0,
            max_value=50,
            value=5,
            step=1,
            help="Total number of open credit accounts"
        )
        
        num_credit_problems = st.number_input(
            "Number of Credit Problems",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="Number of credit problems in your history"
        )
        
        current_credit_balance = st.number_input(
            "Current Credit Balance ($)",
            min_value=0,
            max_value=500000,
            value=5000,
            step=100,
            help="Your current total credit balance"
        )
        
        max_open_credit = st.number_input(
            "Maximum Open Credit ($)",
            min_value=0,
            max_value=1000000,
            value=20000,
            step=1000,
            help="Your maximum available credit limit"
        )
        
        bankruptcies = st.number_input(
            "Number of Bankruptcies",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Number of bankruptcies in your history"
        )
        
        tax_liens = st.number_input(
            "Number of Tax Liens",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Number of tax liens against you"
        )
    
    # Prepare input data
    input_data = {
        'Term': term,
        'Years in current job': years_in_job,
        'Home Ownership': home_ownership,
        'Purpose': purpose,
        'Current Loan Amount': current_loan_amount,
        'Credit Score': credit_score,
        'Annual Income': annual_income,
        'Monthly Debt': monthly_debt,
        'Years of Credit History': years_credit_history,
        'Months since last delinquent': months_since_delinquent,
        'Number of Open Accounts': num_open_accounts,
        'Number of Credit Problems': num_credit_problems,
        'Current Credit Balance': current_credit_balance,
        'Maximum Open Credit': max_open_credit,
        'Bankruptcies': bankruptcies,
        'Tax Liens': tax_liens
    }
    
    # Prediction button
    if st.button("🔍 Predict Loan Status", key="predict_button"):
        with st.spinner("Analyzing your application..."):
            prediction, confidence, probabilities = predict_loan_status(model, feat_engineer, input_data)
            
            if prediction is not None:
                st.markdown("---")
                st.markdown('<h2 class="sub-header">📈 Prediction Results</h2>', unsafe_allow_html=True)
                
                # Create columns for results
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    # Prediction result
                    if prediction == "Fully Paid":
                        st.markdown(f'''
                        <div class="prediction-success">
                            <h3>✅ Loan Approved!</h3>
                            <p><strong>Status:</strong> {prediction}</p>
                            <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="prediction-danger">
                            <h3>❌ Loan Rejected</h3>
                            <p><strong>Status:</strong> {prediction}</p>
                            <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                
                with result_col2:
                    # Confidence gauge
                    gauge_fig = create_gauge_chart(confidence, prediction)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Additional metrics
                st.markdown("### 📊 Risk Assessment")
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    debt_to_income = (monthly_debt * 12) / annual_income if annual_income > 0 else 0
                    st.metric("Debt-to-Income Ratio", f"{debt_to_income:.2%}")
                
                with metric_col2:
                    credit_utilization = current_credit_balance / max_open_credit if max_open_credit > 0 else 0
                    st.metric("Credit Utilization", f"{credit_utilization:.2%}")
                
                with metric_col3:
                    loan_to_income = current_loan_amount / annual_income if annual_income > 0 else 0
                    st.metric("Loan-to-Income Ratio", f"{loan_to_income:.2%}")
                
                with metric_col4:
                    st.metric("Risk Score", f"{(1-confidence)*100:.1f}/100")
                
                # Feature importance chart
                st.markdown("### 📈 Feature Analysis")
                feature_fig = create_feature_importance_chart(feat_engineer, input_data)
                st.plotly_chart(feature_fig, use_container_width=True)
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                if prediction != "Fully Paid":
                    st.markdown("""
                    **To improve your loan approval chances:**
                    - Improve your credit score by paying bills on time
                    - Reduce your debt-to-income ratio
                    - Consider a smaller loan amount
                    - Build more credit history
                    - Reduce existing credit balances
                    """)
                else:
                    st.markdown("""
                    **Congratulations! Your application looks strong:**
                    - Continue maintaining good credit habits
                    - Make payments on time
                    - Keep credit utilization low
                    - Monitor your credit report regularly
                    """)

def about_page():
    st.markdown('<h2 class="sub-header">ℹ️ About LoanScopeAI</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Overview
    
    LoanScopeAI is an advanced machine learning system designed to predict loan eligibility using the state-of-the-art **TabTransformer** architecture. This tool helps financial institutions make faster, more accurate lending decisions while reducing risk.
    
    ## 🧠 Technology Stack
    
    - **Model Architecture**: TabTransformer with self-attention mechanisms
    - **Framework**: PyTorch for deep learning
    - **Frontend**: Streamlit for interactive web interface
    - **Data Processing**: Pandas, NumPy, Scikit-learn
    - **Visualization**: Plotly for interactive charts
    
    ## 🔬 TabTransformer Architecture
    
    The TabTransformer leverages Transformer-based self-attention mechanisms to model relationships among categorical features, offering enhanced performance over traditional methods.
    
    **Key Features:**
    - Contextual embeddings for categorical features
    - Self-attention mechanisms for feature interactions
    - Residual connections for better gradient flow
    - Batch normalization for stable training
    
    ## 📈 Model Performance
    
    Our model has been trained on comprehensive loan data and achieves:
    - High accuracy in loan approval predictions
    - Robust performance across different customer segments
    - Fast inference time for real-time predictions
    
    ## 🔗 References
    
    This implementation is inspired by:
    - **TabTransformer: Tabular Data Modeling Using Contextual Embeddings**
    - Authors: Xin Huang, Ashish Khetan, Milan Cvitkovic, Zohar Karnin
    - Paper: [arXiv:2012.06678](https://arxiv.org/abs/2012.06678)
    
    ## 👥 Contributing
    
    This project is open source and welcomes contributions. Visit our [GitHub repository](https://github.com/aquafish25/LoanScopeAI) to contribute!
    """)

def model_info_page(feat_engineer):
    st.markdown('<h2 class="sub-header">🔧 Model Information</h2>', unsafe_allow_html=True)
    
    # Model architecture info
    st.markdown("### 🏗️ Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Categorical Features:**")
        for i, col in enumerate(feat_engineer.cat_cols, 1):
            num_categories = len(feat_engineer.label_encoders[col].classes_)
            st.write(f"{i}. {col}: {num_categories} categories")
    
    with col2:
        st.markdown("**Selected Continuous Features:**")
        for i, col in enumerate(feat_engineer.selected_cont_cols, 1):
            st.write(f"{i}. {col}")
    
    # Feature engineering pipeline
    st.markdown("### ⚙️ Feature Engineering Pipeline")
    
    pipeline_steps = [
        "Data Loading & Cleaning",
        "Categorical Encoding (Label Encoding)",
        "Feature Selection (Mutual Information)",
        "Scaling (StandardScaler)",
        "Target Encoding"
    ]
    
    for i, step in enumerate(pipeline_steps, 1):
        st.write(f"{i}. {step}")
    
    # Model parameters
    st.markdown("### 📊 Model Parameters")
    
    params_info = {
        "Embedding Dimension": 64,
        "Transformer Depth": 8,
        "Attention Heads": 8,
        "MLP Dropout": 0.3,
        "Selected Features": len(feat_engineer.selected_cont_cols),
        "Categorical Features": len(feat_engineer.cat_cols)
    }
    
    param_col1, param_col2 = st.columns(2)
    
    for i, (param, value) in enumerate(params_info.items()):
        if i % 2 == 0:
            param_col1.metric(param, value)
        else:
            param_col2.metric(param, value)
    
    # Training info
    st.markdown("### 📚 Training Information")
    
    training_info = """
    - **Loss Function**: CrossEntropyLoss
    - **Optimizer**: AdamW with weight decay
    - **Learning Rate Scheduler**: ReduceLROnPlateau
    - **Early Stopping**: Patience of 50 epochs
    - **Gradient Clipping**: Max norm of 1.0
    - **Batch Size**: 512
    - **Maximum Epochs**: 150
    """
    
    st.markdown(training_info)

if __name__ == "__main__":
    main()
