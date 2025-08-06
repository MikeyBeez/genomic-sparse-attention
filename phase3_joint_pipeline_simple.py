#!/usr/bin/env python3
"""
Phase 3: Joint Pipeline Training - Simplified Version

The final test to complete the trilogy with essential comparisons only.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import json

print("🚀 Phase 3: Joint Pipeline Training - The Final Showdown")
print("=" * 80)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create results directory
results_dir = Path('/Users/bard/Code/genomic-sparse-attention/results/phase3')
results_dir.mkdir(parents=True, exist_ok=True)


class JointPipelineModel(nn.Module):
    """Joint Pipeline: Direct task-specific learning without pre-training."""
    
    def __init__(self, vocab_size=6, embed_dim=32, seq_length=200):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        
        # Task-specific embeddings (will be trained 5x faster)
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=5)
        
        # Regulatory pattern detector
        self.pattern_detector = nn.Sequential(
            nn.Conv1d(embed_dim, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Regulatory strength predictor
        self.predictor = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Initialize embeddings
        nn.init.normal_(self.embeddings.weight, mean=0, std=0.01)
    
    def forward(self, x):
        embeddings = self.embeddings(x)
        embeddings = embeddings.transpose(1, 2)  # For Conv1d
        patterns = self.pattern_detector(embeddings).squeeze(-1)
        prediction = self.predictor(patterns).squeeze(-1)
        return prediction


class TraditionalAttentionModel(nn.Module):
    """Traditional attention baseline."""
    
    def __init__(self, vocab_size=6, embed_dim=32, seq_length=200):
        super().__init__()
        
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=5)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        embeddings = self.embeddings(x)
        attended, _ = self.attention(embeddings, embeddings, embeddings)
        pooled = attended.mean(dim=1)
        prediction = self.classifier(pooled).squeeze(-1)
        return prediction


class SparseAttentionModel(nn.Module):
    """Sparse attention model (simplified from Phase 2)."""
    
    def __init__(self, vocab_size=6, embed_dim=32, seq_length=200, sparsity_ratio=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.sparsity_ratio = sparsity_ratio
        
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=5)
        
        self.token_selector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.mlp_approximator = nn.Sequential(
            nn.Linear(embed_dim * 3, 8),
            nn.ReLU(),
            nn.Linear(8, embed_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        embeddings = self.embeddings(x)
        
        # Select sparse tokens
        scores = self.token_selector(embeddings.reshape(-1, self.embed_dim)).reshape(embeddings.shape[0], embeddings.shape[1])
        k = max(1, int(embeddings.shape[1] * self.sparsity_ratio))
        _, top_indices = torch.topk(scores, k, dim=1)
        
        # Gather selected embeddings
        batch_indices = torch.arange(embeddings.shape[0]).unsqueeze(1).expand(-1, k)
        selected_embeddings = embeddings[batch_indices, top_indices]
        
        # Approximate attention
        query = embeddings.mean(dim=1).unsqueeze(1).expand(-1, k, -1)
        qkv_concat = torch.cat([query, selected_embeddings, selected_embeddings], dim=-1)
        processed = self.mlp_approximator(qkv_concat.reshape(-1, self.embed_dim * 3)).reshape(embeddings.shape[0], k, self.embed_dim)
        output = processed.mean(dim=1)
        
        prediction = self.classifier(output).squeeze(-1)
        return prediction


class SyntheticGenomicDataset(Dataset):
    """Generate synthetic genomic sequences with planted regulatory motifs."""
    
    def __init__(self, num_samples=1000, seq_length=200, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.seq_length = seq_length
        self.num_samples = num_samples
        
        # Nucleotide mapping: A=0, T=1, C=2, G=3, N=4, PAD=5
        self.nucleotides = np.array([0, 1, 2, 3])
        
        # Regulatory motifs
        self.regulatory_motifs = {
            'TATAAA': 0.8,
            'CAAT': 0.6,
            'GGGCGG': 0.7,
            'TTGACA': 0.5,
            'TATAAT': 0.6,
        }
        
        self.sequences = []
        self.labels = []
        
        print(f"Generating {num_samples} synthetic sequences...")
        
        for _ in range(num_samples):
            seq, strength = self._generate_sequence()
            self.sequences.append(torch.tensor(seq, dtype=torch.long))
            self.labels.append(torch.tensor(strength, dtype=torch.float))
        
        self.sequences = torch.stack(self.sequences)
        self.labels = torch.stack(self.labels)
        
        print(f"✅ Generated {len(self.sequences)} sequences")
        print(f"   Mean regulatory strength: {self.labels.mean():.3f}")
    
    def _generate_sequence(self):
        sequence = np.random.choice(self.nucleotides, self.seq_length)
        regulatory_strength = 0.1
        
        num_motifs = np.random.poisson(2)
        
        for _ in range(num_motifs):
            motif_item = np.random.choice(list(self.regulatory_motifs.keys()))
            motif_seq = motif_item
            motif_strength = self.regulatory_motifs[motif_item]
            motif_array = np.array([{'A': 0, 'T': 1, 'C': 2, 'G': 3}[nt] for nt in motif_seq])
            
            max_start = self.seq_length - len(motif_array)
            if max_start > 0:
                start_pos = np.random.randint(0, max_start)
                sequence[start_pos:start_pos + len(motif_array)] = motif_array
                regulatory_strength += motif_strength
        
        regulatory_strength = min(regulatory_strength, 1.0)
        regulatory_strength += np.random.normal(0, 0.05)
        regulatory_strength = max(0, min(regulatory_strength, 1.0))
        
        return sequence, regulatory_strength
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class DifferentialOptimizer:
    """Optimizer with different learning rates."""
    
    def __init__(self, model, embedding_lr=0.005, other_lr=0.001):
        self.embedding_optimizer = optim.Adam(model.embeddings.parameters(), lr=embedding_lr)
        
        other_params = []
        for name, param in model.named_parameters():
            if not name.startswith('embeddings'):
                other_params.append(param)
        
        self.other_optimizer = optim.Adam(other_params, lr=other_lr)
        
        print(f"💡 Differential learning rates:")
        print(f"   Embeddings: {embedding_lr:.4f}")
        print(f"   Other layers: {other_lr:.4f}")
    
    def zero_grad(self):
        self.embedding_optimizer.zero_grad()
        self.other_optimizer.zero_grad()
    
    def step(self):
        self.embedding_optimizer.step()
        self.other_optimizer.step()


def train_model(model, train_loader, val_loader, model_name, num_epochs=25, device='cpu', use_differential=False):
    """Train a model."""
    
    model = model.to(device)
    criterion = nn.MSELoss()
    
    if use_differential and hasattr(model, 'embeddings'):
        optimizer = DifferentialOptimizer(model)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    
    print(f"\n🏋️ Training {model_name}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            if use_differential and hasattr(optimizer, 'zero_grad'):
                optimizer.zero_grad()
            else:
                optimizer.zero_grad()
            
            predictions = model(sequences)
            loss = criterion(predictions, labels)
            loss.backward()
            
            if use_differential and hasattr(optimizer, 'step'):
                optimizer.step()
            else:
                optimizer.step()
            
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                predictions = model(sequences)
                val_loss += criterion(predictions, labels).item()
        
        val_losses.append(val_loss / len(val_loader))
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}: Train = {train_losses[-1]:.4f}, Val = {val_losses[-1]:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1]
    }


def evaluate_models(models_results, test_loader, device='cpu'):
    """Evaluate all models on test set."""
    
    print("\n" + "="*80)
    print("🎯 FINAL EVALUATION - THE TRILOGY CONCLUSION")
    print("="*80)
    
    criterion = nn.MSELoss()
    results = {}
    
    for model_name, (model, training_results) in models_results.items():
        model.eval()
        test_loss = 0
        predictions_list = []
        labels_list = []
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                predictions = model(sequences)
                test_loss += criterion(predictions, labels).item()
                
                predictions_list.extend(predictions.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
        
        test_loss /= len(test_loader)
        
        # Calculate metrics
        predictions_array = np.array(predictions_list)
        labels_array = np.array(labels_list)
        
        mae = np.mean(np.abs(predictions_array - labels_array))
        correlation = np.corrcoef(predictions_array, labels_array)[0, 1]
        param_count = sum(p.numel() for p in model.parameters())
        
        results[model_name] = {
            'test_loss': float(test_loss),  # Convert to float for JSON
            'mae': float(mae),
            'correlation': float(correlation),
            'param_count': int(param_count),
            'final_val_loss': float(training_results['final_val_loss'])
        }
        
        print(f"\n📊 {model_name} Results:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   Correlation: {correlation:.3f}")
        print(f"   Parameters: {param_count:,}")
    
    return results


def main():
    """Main Phase 3 experiment."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate dataset
    print("\n🧬 Generating synthetic genomic dataset...")
    full_dataset = SyntheticGenomicDataset(num_samples=1500, seq_length=200, seed=42)
    
    # Split dataset
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"📊 Dataset split: {train_size} train, {val_size} val, {test_size} test")
    
    # Initialize models
    models = {
        'Traditional Attention': TraditionalAttentionModel(),
        'Sparse Attention': SparseAttentionModel(sparsity_ratio=0.1),
        'Joint Pipeline': JointPipelineModel()
    }
    
    # Train all models
    models_results = {}
    
    for model_name, model in models.items():
        use_differential = (model_name == 'Joint Pipeline')
        
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        training_results = train_model(
            model, train_loader, val_loader, model_name, 
            num_epochs=25, device=device, use_differential=use_differential
        )
        
        models_results[model_name] = (model, training_results)
    
    # Final evaluation
    final_results = evaluate_models(models_results, test_loader, device)
    
    # Find winner
    winner = min(final_results.keys(), key=lambda k: final_results[k]['test_loss'])
    winner_metrics = final_results[winner]
    
    # Summary
    print("\n" + "="*80)
    print("🎉 PHASE 3 COMPLETE - TRILOGY CONCLUSION")
    print("="*80)
    
    print(f"\n🏆 TRILOGY WINNER: {winner}")
    print(f"   Final test loss: {winner_metrics['test_loss']:.4f}")
    print(f"   Parameter count: {winner_metrics['param_count']:,}")
    print(f"   Correlation: {winner_metrics['correlation']:.3f}")
    
    # Performance comparison
    baseline_loss = final_results['Traditional Attention']['test_loss']
    print(f"\n📈 PERFORMANCE vs TRADITIONAL ATTENTION:")
    
    for model_name, metrics in final_results.items():
        if model_name != 'Traditional Attention':
            improvement = ((baseline_loss - metrics['test_loss']) / baseline_loss) * 100
            param_efficiency = final_results['Traditional Attention']['param_count'] / metrics['param_count']
            
            print(f"   {model_name}:")
            print(f"     Performance change: {improvement:+.1f}%")
            print(f"     Parameter efficiency: {param_efficiency:.2f}x")
            
            if improvement > 5:
                print(f"     ✅ SIGNIFICANT IMPROVEMENT!")
            elif improvement > 0:
                print(f"     🟡 Modest improvement")
            else:
                print(f"     🔴 Performance loss: {abs(improvement):.1f}%")
    
    print(f"\n🔬 SCIENTIFIC IMPACT SUMMARY:")
    print(f"   Phase 1: ✅ Traditional attention baseline established")
    print(f"   Phase 2: ✅ Sparse approximation confirmed 96.1% efficiency with 10% sparsity")
    print(f"   Phase 3: ✅ Three-way comparison complete")
    
    if winner == 'Joint Pipeline':
        print(f"\n🚀 BREAKTHROUGH: Joint Pipeline beats traditional attention!")
        print(f"   Task-specific learning outperforms general attention")
    elif winner == 'Sparse Attention':
        print(f"\n⚡ EFFICIENCY WINNER: Sparse attention wins!")
        print(f"   Confirms the '90% attention is noise' hypothesis")
    else:
        print(f"\n🤔 Traditional attention remains competitive")
    
    # Save results
    with open(results_dir / 'phase3_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_dir / 'phase3_results.json'}")
    
    return final_results


if __name__ == "__main__":
    final_results = main()
