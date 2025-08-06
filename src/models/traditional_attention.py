"""
Traditional Attention Baseline Model for Genomic Regulatory Prediction

This module implements a standard multi-head attention model for predicting 
regulatory element strength from DNA sequences. This serves as the baseline
for extracting ground truth attention patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class GenomicAttentionBaseline(nn.Module):
    """
    Standard multi-head attention model for genomic regulatory prediction.
    
    Architecture:
    - Nucleotide embeddings (A, T, C, G, N, PAD)
    - Multi-head self-attention 
    - Classification head for regulatory strength
    """
    
    def __init__(
        self, 
        vocab_size: int = 6,  # A, T, C, G, N, PAD
        embed_dim: int = 16,
        num_heads: int = 4,
        seq_length: int = 2000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.seq_length = seq_length
        
        # Nucleotide embeddings
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=5)  # PAD = 5
        
        # Positional encoding (simple learned positions)
        self.pos_encoding = nn.Embedding(seq_length, embed_dim)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Classification head for regulatory strength
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()  # Regulatory strength 0-1
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention extraction.
        
        Args:
            x: Input sequences [batch_size, seq_length]
            
        Returns:
            output: Regulatory strength predictions [batch_size, 1]
            attention_weights: Attention weights [batch_size, num_heads, seq_length, seq_length]
        """
        batch_size, seq_len = x.shape
        
        # Generate position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        
        # Embed nucleotides and add positional encoding
        embedded = self.embeddings(x) + self.pos_encoding(positions)
        embedded = self.dropout(embedded)
        
        # Self-attention with residual connection
        attended, attention_weights = self.attention(embedded, embedded, embedded)
        x1 = self.norm1(embedded + attended)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x1)
        x2 = self.norm2(x1 + ffn_out)
        
        # Global average pooling and classification
        pooled = x2.mean(dim=1)  # [batch_size, embed_dim]
        output = self.classifier(pooled)
        
        return output, attention_weights
    
    def extract_attention_patterns(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract detailed attention patterns for analysis.
        
        Args:
            x: Input sequences [batch_size, seq_length]
            
        Returns:
            Dictionary containing various attention pattern analyses
        """
        with torch.no_grad():
            _, attention_weights = self.forward(x)
            
            # Average attention across heads and batch
            avg_attention = attention_weights.mean(dim=(0, 1))  # [seq_length, seq_length]
            
            # Attention entropy (measure of focus/diffuseness)
            attention_entropy = -torch.sum(
                attention_weights * torch.log(attention_weights + 1e-8), 
                dim=-1
            )
            
            # Row-wise attention sums (which positions attend most)
            attention_focus = attention_weights.sum(dim=-1)  # [batch_size, num_heads, seq_length]
            
            return {
                'raw_attention': attention_weights,
                'average_attention': avg_attention,
                'attention_entropy': attention_entropy,
                'attention_focus': attention_focus
            }


class AttentionGroundTruthExtractor:
    """
    Utility class for extracting and saving attention patterns as ground truth
    for training sparse approximators.
    """
    
    def __init__(self, model: GenomicAttentionBaseline):
        self.model = model
        self.model.eval()
        
    def extract_ground_truth_dataset(
        self, 
        dataloader: torch.utils.data.DataLoader,
        save_path: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extract attention patterns from trained model to create ground truth dataset.
        
        Args:
            dataloader: DataLoader with genomic sequences
            save_path: Optional path to save ground truth data
            
        Returns:
            Dictionary containing input sequences and corresponding attention outputs
        """
        all_sequences = []
        all_predictions = []
        all_attention_weights = []
        all_attention_patterns = []
        
        print("Extracting ground truth attention patterns...")
        
        with torch.no_grad():
            for batch_idx, (sequences, labels) in enumerate(dataloader):
                # Forward pass to get predictions and attention
                predictions, attention_weights = self.model(sequences)
                
                # Extract detailed attention patterns
                attention_patterns = self.model.extract_attention_patterns(sequences)
                
                # Store results
                all_sequences.append(sequences.cpu())
                all_predictions.append(predictions.cpu())
                all_attention_weights.append(attention_weights.cpu())
                all_attention_patterns.append({
                    k: v.cpu() for k, v in attention_patterns.items()
                })
                
                if batch_idx % 10 == 0:
                    print(f"Processed batch {batch_idx}/{len(dataloader)}")
        
        # Concatenate all results
        ground_truth_data = {
            'sequences': torch.cat(all_sequences, dim=0),
            'predictions': torch.cat(all_predictions, dim=0),
            'attention_weights': torch.cat(all_attention_weights, dim=0),
            'average_attention': torch.stack([
                batch['average_attention'] for batch in all_attention_patterns
            ]),
            'attention_entropy': torch.cat([
                batch['attention_entropy'] for batch in all_attention_patterns
            ], dim=0),
            'attention_focus': torch.cat([
                batch['attention_focus'] for batch in all_attention_patterns
            ], dim=0)
        }
        
        # Save if path provided
        if save_path:
            torch.save(ground_truth_data, save_path)
            print(f"Ground truth data saved to {save_path}")
        
        return ground_truth_data
    
    def analyze_attention_quality(self, ground_truth_data: Dict[str, torch.Tensor]) -> Dict:
        """
        Analyze the quality of attention patterns for biological interpretability.
        
        Args:
            ground_truth_data: Ground truth attention patterns
            
        Returns:
            Analysis results including attention statistics and interpretability metrics
        """
        attention_weights = ground_truth_data['attention_weights']
        sequences = ground_truth_data['sequences']
        
        # Calculate attention statistics
        attention_stats = {
            'mean_attention': attention_weights.mean().item(),
            'std_attention': attention_weights.std().item(),
            'max_attention': attention_weights.max().item(),
            'min_attention': attention_weights.min().item()
        }
        
        # Calculate attention sparsity (how concentrated attention is)
        attention_flat = attention_weights.view(-1, attention_weights.shape[-1])
        attention_gini = self._calculate_gini_coefficient(attention_flat)
        
        # Calculate position-wise attention variance (which positions get most attention)
        position_attention = attention_weights.mean(dim=(0, 1))  # Average across batch and heads
        position_variance = position_attention.var(dim=1).mean().item()
        
        analysis_results = {
            'attention_stats': attention_stats,
            'attention_gini': attention_gini.mean().item(),
            'position_variance': position_variance,
            'interpretability_score': self._calculate_interpretability_score(attention_weights)
        }
        
        print("\n=== Attention Quality Analysis ===")
        print(f"Mean attention weight: {attention_stats['mean_attention']:.4f}")
        print(f"Attention concentration (Gini): {analysis_results['attention_gini']:.4f}")
        print(f"Position variance: {position_variance:.4f}")
        print(f"Interpretability score: {analysis_results['interpretability_score']:.4f}")
        
        return analysis_results
    
    def _calculate_gini_coefficient(self, attention: torch.Tensor) -> torch.Tensor:
        """Calculate Gini coefficient for attention concentration."""
        sorted_attention, _ = torch.sort(attention, dim=-1)
        n = attention.shape[-1]
        index = torch.arange(1, n + 1, dtype=torch.float, device=attention.device)
        return ((2 * index - n - 1) * sorted_attention).sum(dim=-1) / (n * sorted_attention.sum(dim=-1))
    
    def _calculate_interpretability_score(self, attention_weights: torch.Tensor) -> float:
        """
        Calculate interpretability score based on attention pattern characteristics.
        Higher score = more interpretable patterns.
        """
        # Attention should be focused (not uniform)
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
        focus_score = 1.0 - (entropy / np.log(attention_weights.shape[-1])).mean()
        
        # Attention should show some structure (not random)
        # This is a simplified metric - could be enhanced with biological knowledge
        structure_score = 0.5  # Placeholder - would analyze attention patterns vs known motifs
        
        return (focus_score + structure_score) / 2
