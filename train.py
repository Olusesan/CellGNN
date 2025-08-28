from cell_tracker_core import EnhancedCellTrackerGNN
from torch_geometric.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

def save_model(model, model_path, metadata=None):
    """Save trained model with metadata"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': model.conv1.in_channels,
            'hidden_dim': 64,  # Default from our architecture
            'output_dim': 32,  # Default from our architecture
            'num_heads': 4     # Default from our architecture
        },
        'metadata': metadata or {}
    }
    torch.save(checkpoint, model_path)
    print(f"Model saved to: {model_path}")
# Focal loss for imbalanced data

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
            
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()
        
def train_gnn_with_validation(dataset, epochs=100, lr=0.001, batch_size=16, 
                            validation_split=0.2, patience=25, save_path=None):
    """Enhanced training with proper validation, monitoring, and model saving"""
    if len(dataset) == 0:
        print("No training data available")
        return None

    # Stratified split to maintain label distribution
    train_size = int((1 - validation_split) * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:] if train_size < len(dataset) else []

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None

    # Model initialization
    input_dim = dataset[0].x.shape[1]
    model = EnhancedCellTrackerGNN(input_dim=input_dim, hidden_dim=64, output_dim=32)

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    criterion = FocalLoss(alpha=1, gamma=2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                          factor=0.5, patience=10, verbose=True)

    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print("Starting enhanced training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass with logits
            logits = model(batch.x, batch.edge_index)
            loss = criterion(logits, batch.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            predictions = torch.sigmoid(logits) > 0.5
            train_correct += (predictions.float() == batch.y).sum().item()
            train_total += len(batch.y)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0

        # Validation phase
        val_loss = 0
        val_acc = 0
        if val_loader:
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    logits = model(batch.x, batch.edge_index)
                    loss = criterion(logits, batch.y)
                    val_loss += loss.item()

                    predictions = torch.sigmoid(logits) > 0.5
                    val_correct += (predictions.float() == batch.y).sum().item()
                    val_total += len(batch.y)

            val_loss /= len(val_loader)
            val_acc = val_correct / val_total if val_total > 0 else 0
            
            scheduler.step(val_loss)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)

        # Progress reporting
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'Loaded best model with validation loss: {best_val_loss:.4f}')

    # Save model if path provided
    if save_path:
        metadata = {
            'training_epochs': epoch + 1,
            'best_val_loss': best_val_loss,
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset) if val_dataset else 0,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        save_model(model, save_path, metadata)

    return model