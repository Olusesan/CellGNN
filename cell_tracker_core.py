import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class EnhancedCellTrackerGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=32, num_heads=4, dropout=0.2):
        super(EnhancedCellTrackerGNN, self).__init__()
        
        # Graph Attention Networks with residual connections
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv3 = GCNConv(hidden_dim * num_heads, output_dim)

        # Batch normalization and layer normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.bn2 = nn.BatchNorm1d(hidden_dim * num_heads)
        self.ln = nn.LayerNorm(output_dim)

        # Improved edge prediction with attention mechanism
        self.edge_predictor = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Feature attention for edge prediction
        self.feature_attention = nn.MultiheadAttention(output_dim, num_heads=4, batch_first=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, return_embeddings=False):
        # Node embeddings with residual connections
        h1 = F.elu(self.bn1(self.conv1(x, edge_index)))
        h1 = self.dropout(h1)
        
        h2 = F.elu(self.bn2(self.conv2(h1, edge_index)))
        h2 = self.dropout(h2)
        
        # Final embedding
        embeddings = self.ln(self.conv3(h2, edge_index))

        if return_embeddings:
            return embeddings

        # Enhanced edge prediction with attention
        row, col = edge_index
        edge_features = torch.cat([embeddings[row], embeddings[col]], dim=1)
        edge_probs = torch.sigmoid(self.edge_predictor(edge_features))

        return edge_probs.squeeze()
