import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

class TabularDataset(Dataset):
    """PyTorch Dataset for handling tabular data with categorical and continuous features.
    
    Args:
        cat_data: Categorical feature matrix
        cont_data: Continuous feature matrix
        targets: Target labels
    """
    def __init__(self, cat_data: np.ndarray, cont_data: np.ndarray, targets: np.ndarray):
        self.cat_features = torch.tensor(cat_data, dtype=torch.long)
        self.cont_features = torch.tensor(cont_data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return (
            self.cat_features[idx],
            self.cont_features[idx],
            self.targets[idx]
        )

class FeatureEngineering:
    """Feature engineering pipeline for preprocessing and feature selection.
    
    Args:
        cat_cols: List of categorical column names
        cont_cols: List of continuous column names
        target_col: Name of target column
        k_features: Number of top features to select
    """
    def __init__(self,
                 cat_cols: list[str],
                 cont_cols: list[str],
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
    
    def load_and_prepare_data(self, csv_path: str) -> pd.DataFrame:
        """Load data from CSV and handle missing values."""
        df = pd.read_csv(csv_path, low_memory=False)
        df = self.clean_numeric_cols(df)
        df = df.dropna()
        print(f"[Data Loaded] Samples: {df.shape[0]}, Features: {df.shape[1]}")
        print(f"[Missing Values] Final dataset shape: {df.shape}")
        return df
    
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
                        lambda x: x if x in known_categories else 'UNKNOWN'
                    )
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
        return df_encoded
    
    def select_features(self, df: pd.DataFrame, fit: bool = True) -> list[str]:
        """Select top k continuous features using mutual information."""
        available_cols = [col for col in self.cont_cols if col in df.columns]
        if fit:
            X = df[available_cols].values
            y = df[self.target_col].values
            X = np.nan_to_num(X, nan=0.0)
            k = min(self.k_features, len(available_cols))
            self.selector = SelectKBest(score_func=mutual_info_classif, k=k)
            self.selector.fit(X, y)
            self.selected_cont_cols = [
                available_cols[i]
                for i in self.selector.get_support(indices=True)
            ]
            print(f"[Feature Selection] Top {k} features:")
            for i, feature in enumerate(self.selected_cont_cols, 1):
                print(f"  {i}. {feature}")
        return self.selected_cont_cols
    
    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit preprocessing pipeline and transform training data."""
        print("\n" + "="*50)
        print("FITTING FEATURE ENGINEERING PIPELINE")
        print("="*50)
        
        df_encoded = self.encode_categoricals(df, fit=True)
        y = self.target_encoder.fit_transform(df_encoded[self.target_col])
        self.select_features(df_encoded, fit=True)
        X_cat = df_encoded[self.cat_cols].values.astype(np.int64)
        X_cont = self.scaler.fit_transform(df_encoded[self.selected_cont_cols].values).astype(np.float32)
        self.is_fitted = True
        return X_cat, X_cont, y.astype(np.int64)
    
    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform data using fitted pipeline."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit_transform first")
        df_encoded = self.encode_categoricals(df, fit=False)
        if self.target_col in df_encoded.columns:
            y = self.target_encoder.transform(df_encoded[self.target_col])
        else:
            y = np.array([])
        X_cat = df_encoded[self.cat_cols].values.astype(np.int64)
        X_cont = self.scaler.transform(df_encoded[self.selected_cont_cols].values).astype(np.float32)
        return X_cat, X_cont, y.astype(np.int64)
    
    def save_pipeline(self, output_dir: str = 'feature_config'):
        """Save feature engineering pipeline to disk."""
        os.makedirs(output_dir, exist_ok=True)
        pipeline_path = os.path.join(output_dir, 'feature_pipeline.pkl')
        with open(pipeline_path, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"[Pipeline Saved] Path: {pipeline_path}")
    
    @classmethod
    def load_pipeline(cls, pipeline_path: str):
        """Load feature engineering pipeline from disk."""
        with open(pipeline_path, 'rb') as f:
            data = pickle.load(f)
        pipeline = cls([], [], '')
        pipeline.__dict__.update(data)
        return pipeline

class TabTransformer(nn.Module):
    """TabTransformer model architecture for tabular data.
    
    Args:
        cat_cardinalities: Dictionary of categorical feature cardinalities
        num_cont_features: Number of continuous features
        num_classes: Number of output classes
        dim: Embedding dimension
        depth: Number of transformer layers
        heads: Number of attention heads
        mlp_dropout: Dropout rate for MLP
    """
    def __init__(self,
                 cat_cardinalities: dict[str, int],
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

class TabTransformerTrainer:
    """Training and evaluation pipeline for TabTransformer model."""
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 150,
              lr: float = 1e-4,
              weight_decay: float = 1e-5,
              patience: int = 50):
        """Train model with early stopping and learning rate scheduling."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        criterion = nn.CrossEntropyLoss()
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training header
        print(f"\n{'='*50}")
        print(f"Training Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*50}\n")
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss, correct_preds, total_samples = 0, 0, 0
            
            for batch in train_loader:
                cat, cont, targets = batch
                cat, cont, targets = cat.to(self.device), cont.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(cat, cont)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_samples += targets.size(0)
                correct_preds += (predicted == targets).sum().item()
            
            # Validation phase
            val_loss, val_correct, val_total = self.evaluate(val_loader, criterion)
            avg_train_loss = epoch_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            scheduler.step(avg_val_loss)
            
            # Epoch summary
            print(f"Epoch {epoch+1:03d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Acc: {val_acc:.2f}% | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[Early Stopping] Triggered at epoch {epoch+1}")
                    break
        
        # Final evaluation
        self.model.load_state_dict(torch.load('best_model.pth'))
        final_val_loss, final_val_correct, final_val_total = self.evaluate(val_loader, criterion)
        final_val_acc = 100 * final_val_correct / final_val_total
        print(f"\n[Training Complete] Best validation accuracy: {final_val_acc:.2f}%")
    
    def evaluate(self, data_loader: DataLoader, criterion) -> tuple[float, int, int]:
        """Evaluate model on given data loader."""
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for batch in data_loader:
                cat, cont, targets = batch
                cat, cont, targets = cat.to(self.device), cont.to(self.device), targets.to(self.device)
                
                outputs = self.model(cat, cont)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return total_loss, correct, total
    
    def save(self, path: str):
        """Save model weights to file."""
        torch.save(self.model.state_dict(), path)
        print(f"[Model Saved] Path: {path}")

def main(model_save_path: str = 'tabtransformer.pth',
         feature_pipeline_dir: str = 'features'):
    """Main training pipeline."""
    CAT_COLS = ['Term', 'Years in current job', 'Home Ownership', 'Purpose']
    CONT_COLS = [
        'Current Loan Amount', 'Credit Score', 'Annual Income', 'Monthly Debt',
        'Years of Credit History', 'Months since last delinquent',
        'Number of Open Accounts', 'Number of Credit Problems',
        'Current Credit Balance', 'Maximum Open Credit', 'Bankruptcies', 'Tax Liens'
    ]
    TARGET_COL = 'Loan Status'
    
    # Initialize feature engineering pipeline
    feat_engineer = FeatureEngineering(
        cat_cols=CAT_COLS,
        cont_cols=CONT_COLS,
        target_col=TARGET_COL,
        k_features=8
    )
    
    # Load and preprocess data
    train_df = feat_engineer.load_and_prepare_data('data/train.csv')
    X_cat, X_cont, y = feat_engineer.fit_transform(train_df)
    
    # Create datasets
    train_idx, val_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y)
    train_ds = TabularDataset(X_cat[train_idx], X_cont[train_idx], y[train_idx])
    val_ds = TabularDataset(X_cat[val_idx], X_cont[val_idx], y[val_idx])
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512)
    
    # Initialize model
    cat_cardinalities = {
        feat: len(feat_engineer.label_encoders[feat].classes_)
        for feat in CAT_COLS
    }
    
    model = TabTransformer(
        cat_cardinalities=cat_cardinalities,
        num_cont_features=len(feat_engineer.selected_cont_cols),
        num_classes=len(feat_engineer.target_encoder.classes_),
        dim=64,
        depth=8,
        mlp_dropout=0.3
    )
    
    # Train model
    trainer = TabTransformerTrainer(model)
    trainer.train(
        train_loader, 
        val_loader, 
        epochs=150,
        weight_decay=1e-5
    )
    
    # Save
    trainer.save(model_save_path)
    feat_engineer.save_pipeline(feature_pipeline_dir)

if __name__ == "__main__":
    main(
        model_save_path='models/loan_model.pth',
        feature_pipeline_dir='configs/feature_pipeline'
    )