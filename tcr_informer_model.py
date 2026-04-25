"""
TCR-Informer Hybrid Model
Combines T-Cell Receptor adaptive selection with Informer's ProbSparse Attention
for efficient, large-scale stock price forecasting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class ProbSparseAttention(nn.Module):
    """
    Probabilistic Sparse Attention inspired by Informer
    Reduces complexity from O(L²) to O(L log L)
    """
    
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbSparseAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    
    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        Probabilistic selection of Top-k queries
        Q: [B, H, L_Q, D]
        K: [B, H, L_K, D]
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        
        # Sample K randomly
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=Q.device)
        K_sample = K_expand[:, :, torch.arange(L_Q, device=Q.device).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)
        
        # Find top-k with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]
        
        # Calculate Q_K for top queries
        Q_reduce = Q[torch.arange(B, device=Q.device)[:, None, None],
                     torch.arange(H, device=Q.device)[None, :, None],
                     M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        
        return Q_K, M_top
    
    def _get_initial_context(self, V, L_Q):
        """Initialize context vector"""
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            context = V.cumsum(dim=-2)
        return context
    
    def forward(self, queries, keys, values, attn_mask=None):
        """
        queries: [B, L_Q, H, D]
        keys: [B, L_K, H, D]
        values: [B, L_V, H, D]
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        
        queries = queries.transpose(1, 2)  # [B, H, L_Q, D]
        keys = keys.transpose(1, 2)        # [B, H, L_K, D]
        values = values.transpose(1, 2)    # [B, H, L_V, D]
        
        U_part = self.factor * int(np.ceil(np.log(L_K)))
        u = self.factor * int(np.ceil(np.log(L_Q)))
        
        U_part = min(U_part, L_K)
        u = min(u, L_Q)
        
        # Sparse attention
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        
        scale = self.scale or 1. / np.sqrt(D)
        scores_top = scores_top * scale
        
        # Get context and update
        context = self._get_initial_context(values, L_Q)
        attn = torch.softmax(scores_top, dim=-1)
        context[torch.arange(B, device=queries.device)[:, None, None],
                torch.arange(H, device=queries.device)[None, :, None],
                index, :] = torch.matmul(attn, values)
        
        context = context.transpose(1, 2).contiguous()
        
        if self.output_attention:
            return context, attn
        else:
            return context, None


class TCRAttentionLayer(nn.Module):
    """
    TCR-Attention Layer with Prob Sparse Attention
    Combines adaptive selection with sparse attention
    """
    
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, dropout=0.1):
        super(TCRAttentionLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        
        return self.out_projection(out), attn


class TemporalEmbedding(nn.Module):
    """
    Multi-scale temporal feature embedding
    Captures daily, weekly, monthly patterns
    """
    
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TemporalEmbedding, self).__init__()
        self.d_model = d_model
        self.embed_type = embed_type
        self.freq = freq
        
        if embed_type == 'timeF':
            self.minute_embed = nn.Embedding(60, d_model // 4)
            self.hour_embed = nn.Embedding(24, d_model // 4)
            self.day_embed = nn.Embedding(32, d_model // 4)
            self.month_embed = nn.Embedding(13, d_model // 4)
    
    def forward(self, x):
        # x shape: [B, L, 1]
        # For simplicity, return positional encoding
        batch_size, seq_len, _ = x.shape
        
        if self.embed_type == 'timeF':
            # Create temporal features
            positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
            minute = (positions % 60).unsqueeze(0).expand(batch_size, -1)
            hour = ((positions // 60) % 24).unsqueeze(0).expand(batch_size, -1)
            day = ((positions // (60*24)) % 32).unsqueeze(0).expand(batch_size, -1)
            month = ((positions // (60*24*32)) % 13).unsqueeze(0).expand(batch_size, -1)
            
            temporal_embed = torch.cat([
                self.minute_embed(minute),
                self.hour_embed(hour),
                self.day_embed(day),
                self.month_embed(month)
            ], dim=-1)
            return temporal_embed
        else:
            # Positional encoding
            pos_enc = self._positional_encoding(seq_len, self.d_model, x.device)
            return pos_enc.unsqueeze(0).expand(batch_size, -1, -1)
    
    def _positional_encoding(self, seq_len, d_model, device):
        position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * -(np.log(10000.0) / d_model))
        
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


class TCREncoderLayer(nn.Module):
    """
    TCR Encoder with Temporal Embedding
    """
    
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='gelu'):
        super(TCREncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == 'gelu' else F.relu
    
    def forward(self, x, attn_mask=None):
        # Self-attention
        new_x, attn = self.attention(x, x, x, attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)
        
        # Feed-forward
        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y), attn


class TCRInformerEncoder(nn.Module):
    """
    Multi-layer TCR-Informer Encoder
    """
    
    def __init__(self, layers, distil=True):
        super(TCRInformerEncoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.distil = distil
    
    def forward(self, x, attn_mask=None):
        attns = []
        for i, layer in enumerate(self.layers):
            x, attn = layer(x, attn_mask)
            attns.append(attn)
            
            # Distillation: reduce sequence length by half
            if self.distil and i < len(self.layers) - 1:
                x = x[:, ::2, :]  # Keep every other timestep
        
        return x, attns


class TCRInformerDecoder(nn.Module):
    """
    TCR-Informer Decoder for autoregressive forecasting
    """
    
    def __init__(self, layers):
        super(TCRInformerDecoder, self).__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, encoder_output, attn_mask=None):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, encoder_output, encoder_output, attn_mask)
            attns.append(attn)
        
        return x, attns


class TCRInformer(nn.Module):
    """
    Complete TCR-Informer Model
    Combines:
    - Probabilistic Sparse Attention (Informer)
    - Adaptive Regressor Selection (TCR)
    - Temporal Embeddings
    - Multi-scale Feature Extraction
    """
    
    def __init__(self,
                 enc_in,        # Input dimensions
                 dec_in,        # Decoder input dimensions
                 c_out,         # Output dimensions
                 seq_len,       # Input sequence length
                 label_len,     # Label sequence length
                 pred_len,      # Prediction length
                 d_model=512,
                 n_heads=8,
                 e_layers=2,
                 d_layers=1,
                 d_ff=2048,
                 dropout=0.1,
                 attn='prob',
                 embed_type='timeF',
                 freq='h',
                 activation='gelu',
                 distil=True,
                 output_attention=False):
        super(TCRInformer, self).__init__()
        
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.c_out = c_out
        
        # Input embedding
        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.dec_embedding = nn.Linear(dec_in, d_model)
        
        # Temporal embedding
        self.temporal_embed = TemporalEmbedding(d_model, embed_type, freq)
        
        # Encoder
        attn_cls = ProbSparseAttention if attn == 'prob' else nn.MultiheadAttention
        encoder_layers = []
        for _ in range(e_layers):
            encoder_layers.append(
                TCREncoderLayer(
                    TCRAttentionLayer(
                        ProbSparseAttention(mask_flag=True, factor=5, output_attention=output_attention),
                        d_model, n_heads, dropout=dropout
                    ),
                    d_model, d_ff, dropout, activation
                )
            )
        self.encoder = TCRInformerEncoder(encoder_layers, distil=distil)
        
        # Decoder
        decoder_layers = []
        for _ in range(d_layers):
            decoder_layers.append(
                TCREncoderLayer(
                    TCRAttentionLayer(
                        ProbSparseAttention(mask_flag=True, factor=5, output_attention=output_attention),
                        d_model, n_heads, dropout=dropout
                    ),
                    d_model, d_ff, dropout, activation
                )
            )
        self.decoder = TCRInformerDecoder(decoder_layers)
        
        # Output projection
        self.projection = nn.Linear(d_model, c_out)
    
    def forward(self, x_enc, x_dec):
        """
        x_enc: [B, seq_len, enc_in]
        x_dec: [B, label_len + pred_len, dec_in]
        """
        # Encoder
        enc_out = self.enc_embedding(x_enc)
        enc_out = enc_out + self.temporal_embed(x_enc[:, :, :1])  # Add temporal features
        
        enc_out, enc_attn = self.encoder(enc_out)
        
        # Decoder
        dec_out = self.dec_embedding(x_dec)
        dec_out = dec_out + self.temporal_embed(x_dec[:, :, :1])
        
        # Cross-attention with encoder output
        dec_out, dec_attn = self.decoder(dec_out, enc_out)
        
        # Output projection
        out = self.projection(dec_out)
        
        if self.output_attention:
            return out, enc_attn, dec_attn
        else:
            return out


class TCRInformerForecaster:
    """
    High-level wrapper for TCR-Informer forecasting
    """
    
    def __init__(self,
                 seq_len=96,
                 label_len=48,
                 pred_len=24,
                 d_model=512,
                 n_heads=8,
                 e_layers=2,
                 d_layers=1,
                 d_ff=2048,
                 dropout=0.1,
                 learning_rate=0.0001,
                 epochs=100,
                 batch_size=32,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 verbose=True):
        
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.verbose = verbose
        
        self.model = None
        self.optimizer = None
        self.scaler = StandardScaler()
        self.criterion = nn.MSELoss()
        self.training_losses = []
        self.validation_losses = []
    
    def build_model(self, enc_in, dec_in, c_out):
        """Build TCR-Informer model"""
        self.model = TCRInformer(
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=c_out,
            seq_len=self.seq_len,
            label_len=self.label_len,
            pred_len=self.pred_len,
            d_model=512,
            n_heads=8,
            e_layers=2,
            d_layers=1,
            d_ff=2048,
            dropout=0.1,
            attn='prob',
            embed_type='timeF',
            distil=True
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        if self.verbose:
            print(f"Model built on device: {self.device}")
            print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data(self, data, train_ratio=0.8):
        """
        Prepare data for training
        data: numpy array [N, features]
        """
        # Normalize
        data_normalized = self.scaler.fit_transform(data)
        
        # Create sequences
        sequences_x = []
        sequences_y = []
        
        for i in range(len(data_normalized) - self.seq_len - self.pred_len + 1):
            # Encoder input
            x_enc = data_normalized[i:i + self.seq_len]
            
            # Decoder input: label + future
            x_dec = data_normalized[i + self.seq_len - self.label_len:i + self.seq_len + self.pred_len]
            
            # Target: future values
            y = data_normalized[i + self.seq_len:i + self.seq_len + self.pred_len]
            
            sequences_x.append((x_enc, x_dec))
            sequences_y.append(y)
        
        sequences_x = np.array(sequences_x)
        sequences_y = np.array(sequences_y)
        
        # Train-test split
        split_idx = int(len(sequences_x) * train_ratio)
        
        train_x = (torch.FloatTensor(sequences_x[:split_idx, 0]).to(self.device),
                  torch.FloatTensor(sequences_x[:split_idx, 1]).to(self.device))
        train_y = torch.FloatTensor(sequences_y[:split_idx]).to(self.device)
        
        test_x = (torch.FloatTensor(sequences_x[split_idx:, 0]).to(self.device),
                 torch.FloatTensor(sequences_x[split_idx:, 1]).to(self.device))
        test_y = torch.FloatTensor(sequences_y[split_idx:]).to(self.device)
        
        if self.verbose:
            print(f"Training samples: {len(train_x[0])}, Testing samples: {len(test_x[0])}")
        
        return train_x, train_y, test_x, test_y
    
    def train(self, train_x, train_y, val_x=None, val_y=None):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            
            for batch_idx in range(0, len(train_x[0]), self.batch_size):
                batch_x_enc = train_x[0][batch_idx:batch_idx + self.batch_size]
                batch_x_dec = train_x[1][batch_idx:batch_idx + self.batch_size]
                batch_y = train_y[batch_idx:batch_idx + self.batch_size]
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(batch_x_enc, batch_x_dec)
                
                # Loss
                loss = self.criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_epoch_loss = epoch_loss / n_batches
            self.training_losses.append(avg_epoch_loss)
            
            # Validation
            if val_x is not None:
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(val_x[0], val_x[1])
                    val_loss = self.criterion(val_predictions, val_y).item()
                    self.validation_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break
                
                if self.verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {avg_epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if self.verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {avg_epoch_loss:.6f}")
    
    def predict(self, test_x, test_y):
        """Make predictions on test data"""
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(test_x[0], test_x[1])
        
        # Denormalize
        predictions_np = predictions.cpu().numpy()
        test_y_np = test_y.cpu().numpy()
        
        predictions_denorm = self.scaler.inverse_transform(
            np.column_stack([predictions_np, np.zeros((predictions_np.shape[0], self.scaler.n_features_in_ - 1))])
        )[:, :predictions_np.shape[1]]
        
        test_y_denorm = self.scaler.inverse_transform(
            np.column_stack([test_y_np, np.zeros((test_y_np.shape[0], self.scaler.n_features_in_ - 1))])
        )[:, :test_y_np.shape[1]]
        
        return predictions_denorm, test_y_denorm
    
    def save(self, path):
        """Save model"""
        torch.save(self.model.state_dict(), path)
        if self.verbose:
            print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        if self.verbose:
            print(f"Model loaded from {path}")
