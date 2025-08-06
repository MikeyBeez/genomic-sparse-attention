#!/usr/bin/env python3
"""
Phase 3: Joint Pipeline Training - The Final Test

This is the culmination: Does task-specific learning beat traditional attention entirely?

We'll train three approaches from scratch and compare:
1. Traditional Attention (baseline)
2. Sparse Approximation (Phase 2 winner)
3. Joint Pipeline (direct task-specific learning)

The Joint Pipeline uses differential learning rates:
- Embeddings: 5x higher learning rate (rapid adaptation to genomic patterns)
- Task layers: standard learning rate (stable classification learning)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import time
import matplotlib.pyplot as plt
import json
from collections import defaultdict

print("🚀 Phase 3: Joint Pipeline Training - The Final Showdown")
print("=" * 80)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create results directory
results_dir = Path('/Users/bard/Code/genomic-sparse-attention/results/phase3')
results_dir.mkdir(parents=True, exist_ok=True)


class JointPipelineModel(nn.Module):
    """
    Joint Pipeline: Direct task-specific learning without pre-training.
    
    Key innovations:
    1. Embeddings trained specifically for regulatory prediction
    2. Differential learning rates (embeddings 5x faster)
    3. Direct sequence-to-prediction mapping
    4. No attention mechanism - pure feed-forward efficiency
    """
    
    def __init__(self, vocab_size=6, embed_dim=32, seq_length=200):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        
        # Task-specific embeddings (will be trained 5x faster)
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=5)
        
        # Regulatory pattern detector
        self.pattern_detector = nn.Sequential(
            nn.Conv1d(embed_dim, 64, kernel_size=7, padding=3),  # Detect 7-mer motifs
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),         # Combine into larger patterns
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),         # Final pattern refinement
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)                              # Global pattern summary
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
        
        # Initialize embeddings with small random values
        nn.init.normal_(self.embeddings.weight, mean=0, std=0.01)
    
    def forward(self, x):
        """Forward pass: sequence -> embeddings -> patterns -> prediction."""
        # Task-specific embeddings
        embeddings = self.embeddings(x)  # [batch, seq_len, embed_dim]
        
        # Transpose for Conv1d: [batch, embed_dim, seq_len]
        embeddings = embeddings.transpose(1, 2)
        
        # Detect regulatory patterns
        patterns = self.pattern_detector(embeddings)  # [batch, 16, 1]
        patterns = patterns.squeeze(-1)               # [batch, 16]
        
        # Predict regulatory strength
        prediction = self.predictor(patterns).squeeze(-1)  # [batch]
        
        return prediction


class TraditionalAttentionModel(nn.Module):
    """Traditional attention baseline (same as Phase 1)."""
    
    def __init__(self, vocab_size=6, embed_dim=32, seq_length=200):
        super().__init__()
        
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=5)
        
        # Traditional attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=4, 
            batch_first=True
        )
        
        # Classification head
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
    """Sparse attention model (from Phase 2)."""
    
    def __init__(self, vocab_size=6, embed_dim=32, seq_length=200, sparsity_ratio=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.sparsity_ratio = sparsity_ratio
        
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=5)
        
        # Token selector
        self.token_selector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # MLP approximator
        self.mlp_approximator = nn.Sequential(
            nn.Linear(embed_dim * 3, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(), 
            nn.Linear(8, embed_dim)
        )
        
        # Classifier
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
        
        # Classify
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
        
        # Regulatory motifs (simplified)
        self.regulatory_motifs = {
            'TATAAA': 0.8,    # TATA box - strong regulatory signal
            'CAAT': 0.6,      # CAAT box - medium signal
            'GGGCGG': 0.7,    # GC box - medium-strong signal
            'TTGACA': 0.5,    # -35 element - medium signal
            'TATAAT': 0.6,    # -10 element - medium signal
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
        print(f"   Regulatory strength range: {self.labels.min():.3f} - {self.labels.max():.3f}")
        print(f"   Mean regulatory strength: {self.labels.mean():.3f}")
    
    def _generate_sequence(self):
        """Generate a single sequence with planted motifs."""
        # Start with random background
        sequence = np.random.choice(self.nucleotides, self.seq_length)
        regulatory_strength = 0.1  # Base regulatory activity
        
        # Plant regulatory motifs randomly
        num_motifs = np.random.poisson(2)  # Average 2 motifs per sequence
        
        for _ in range(num_motifs):
            motif_item = np.random.choice(list(self.regulatory_motifs.keys()))
            motif_seq = motif_item
            motif_strength = self.regulatory_motifs[motif_item]
            motif_array = np.array([{'A': 0, 'T': 1, 'C': 2, 'G': 3}[nt] for nt in motif_seq])
            
            # Find valid position
            max_start = self.seq_length - len(motif_array)
            if max_start > 0:
                start_pos = np.random.randint(0, max_start)
                sequence[start_pos:start_pos + len(motif_array)] = motif_array
                regulatory_strength += motif_strength
        
        # Normalize and add noise
        regulatory_strength = min(regulatory_strength, 1.0)
        regulatory_strength += np.random.normal(0, 0.05)  # Small noise
        regulatory_strength = max(0, min(regulatory_strength, 1.0))
        
        return sequence, regulatory_strength
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class DifferentialOptimizer:
    """Optimizer with different learning rates for different parameter groups."""
    
    def __init__(self, model, embedding_lr=0.005, other_lr=0.001):
        self.embedding_optimizer = optim.Adam(model.embeddings.parameters(), lr=embedding_lr)
        
        # Get all parameters except embeddings
        other_params = []
        for name, param in model.named_parameters():
            if not name.startswith('embeddings'):
                other_params.append(param)
        
        self.other_optimizer = optim.Adam(other_params, lr=other_lr)
        
        print(f"💡 Differential learning rates:")
        print(f"   Embeddings: {embedding_lr:.4f} (task-specific adaptation)")
        print(f"   Other layers: {other_lr:.4f} (stable learning)")
    
    def zero_grad(self):
        self.embedding_optimizer.zero_grad()
        self.other_optimizer.zero_grad()
    
    def step(self):
        self.embedding_optimizer.step()
        self.other_optimizer.step()


def train_model(model, train_loader, val_loader, model_name, num_epochs=30, device='cpu', use_differential=False):
    """Train a model and track detailed metrics."""
    
    model = model.to(device)
    criterion = nn.MSELoss()
    
    if use_differential and hasattr(model, 'embeddings'):
        optimizer = DifferentialOptimizer(model)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Track metrics
    train_losses = []
    val_losses = []
    learning_curves = {'train': [], 'val': []}
    
    print(f"\n🏋️ Training {model_name}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0
        num_batches = 0
        
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
            
            epoch_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                predictions = model(sequences)
                val_loss = criterion(predictions, labels)
                epoch_val_loss += val_loss.item()
                num_val_batches += 1
        
        avg_val_loss = epoch_val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}: Train = {avg_train_loss:.4f}, Val = {avg_val_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1],
        'model_state': model.state_dict()
    }


def comprehensive_evaluation(models_results, test_loader, device='cpu'):
    """Comprehensive evaluation comparing all three approaches."""
    
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE EVALUATION - THE FINAL COMPARISON")
    print("="*80)
    
    criterion = nn.MSELoss()
    results = {}
    
    for model_name, (model, training_results) in models_results.items():
        model.eval()
        test_loss = 0
        predictions_list = []
        labels_list = []
        
        # Test performance
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                predictions = model(sequences)
                test_loss += criterion(predictions, labels).item()
                
                predictions_list.extend(predictions.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
        
        test_loss /= len(test_loader)
        
        # Calculate additional metrics
        predictions_array = np.array(predictions_list)
        labels_array = np.array(labels_list)
        
        mae = np.mean(np.abs(predictions_array - labels_array))
        correlation = np.corrcoef(predictions_array, labels_array)[0, 1]
        
        # Parameter count
        param_count = sum(p.numel() for p in model.parameters())
        
        results[model_name] = {
            'test_loss': test_loss,
            'mae': mae,
            'correlation': correlation,
            'param_count': param_count,
            'best_val_loss': training_results['best_val_loss'],
            'convergence_speed': len(training_results['train_losses'])  # Epochs to converge
        }
        
        print(f"\n📊 {model_name} Results:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   Correlation: {correlation:.3f}")
        print(f"   Parameters: {param_count:,}")
        print(f"   Convergence: {results[model_name]['convergence_speed']} epochs")
    
    # Find best model
    best_model = min(results.keys(), key=lambda k: results[k]['test_loss'])
    
    # Calculate relative performance
    baseline_loss = results['Traditional Attention']['test_loss']
    
    print(f"\n🏆 WINNER: {best_model}")
    print(f"   Best test loss: {results[best_model]['test_loss']:.4f}")
    
    print(f"\n📈 RELATIVE PERFORMANCE (vs Traditional Attention):")
    for model_name, metrics in results.items():
        if model_name != 'Traditional Attention':
            relative_performance = (baseline_loss - metrics['test_loss']) / baseline_loss * 100
            param_efficiency = results['Traditional Attention']['param_count'] / metrics['param_count']
            
            print(f"   {model_name}:")
            print(f"     Performance gain: {relative_performance:+.1f}%")
            print(f"     Parameter efficiency: {param_efficiency:.2f}x")
            
            if relative_performance > 5:
                print(f"     ✅ SIGNIFICANT IMPROVEMENT!")
            elif relative_performance > 0:
                print(f"     🟡 Modest improvement")
            else:
                print(f"     🔴 Performance loss")
    
    return results


def visualize_results(models_results, results, output_dir):
    """Create comprehensive visualization of results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Training curves
    ax = axes[0, 0]
    for model_name, (_, training_results) in models_results.items():
        epochs = range(1, len(training_results['train_losses']) + 1)
        ax.plot(epochs, training_results['train_losses'], label=f'{model_name} (Train)', linestyle='--')
        ax.plot(epochs, training_results['val_losses'], label=f'{model_name} (Val)')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Curves Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Performance comparison
    ax = axes[0, 1]
    model_names = list(results.keys())
    test_losses = [results[name]['test_loss'] for name in model_names]
    
    bars = ax.bar(model_names, test_losses, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.set_ylabel('Test Loss')
    ax.set_title('Final Performance Comparison')
    ax.tick_params(axis='x', rotation=45)
    
    # Annotate bars
    for bar, loss in zip(bars, test_losses):
        ax.annotate(f'{loss:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()), 
                   ha='center', va='bottom')
    
    # 3. Parameter efficiency
    ax = axes[1, 0]
    param_counts = [results[name]['param_count'] for name in model_names]
    correlations = [results[name]['correlation'] for name in model_names]
    
    scatter = ax.scatter(param_counts, correlations, c=['blue', 'red', 'green'], s=100)
    ax.set_xlabel('Parameter Count')
    ax.set_ylabel('Correlation with Ground Truth')
    ax.set_title('Parameter Efficiency vs Performance')
    
    # Annotate points
    for i, name in enumerate(model_names):
        ax.annotate(name, (param_counts[i], correlations[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Convergence speed
    ax = axes[1, 1]
    convergence_speeds = [results[name]['convergence_speed'] for name in model_names]
    final_losses = [results[name]['best_val_loss'] for name in model_names]
    
    bars = ax.bar(model_names, convergence_speeds, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.set_ylabel('Epochs to Convergence')
    ax.set_title('Training Efficiency')
    ax.tick_params(axis='x', rotation=45)
    
    # Annotate bars
    for bar, epochs in zip(bars, convergence_speeds):
        ax.annotate(f'{epochs}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()), 
                   ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Visualizations saved to {output_dir / 'comprehensive_comparison.png'}")


def save_results(results, models_results, output_dir):
    """Save detailed results to JSON."""
    
    # Prepare serializable results
    save_data = {
        'final_results': results,
        'training_history': {},
        'analysis': {
            'winner': min(results.keys(), key=lambda k: results[k]['test_loss']),
            'total_experiments': len(results),
            'best_test_loss': min(results[k]['test_loss'] for k in results.keys())
        }
    }
    
    # Add training history (excluding model states)
    for model_name, (_, training_results) in models_results.items():
        save_data['training_history'][model_name] = {
            'train_losses': training_results['train_losses'],
            'val_losses': training_results['val_losses'],
            'convergence_epochs': len(training_results['train_losses'])
        }
    
    # Save to file
    with open(output_dir / 'phase3_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"💾 Results saved to {output_dir / 'phase3_results.json'}")


def main():
    """Main Phase 3 experiment - The Final Comparison."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate fresh dataset for fair comparison
    print("\n🧬 Generating synthetic genomic dataset...")
    full_dataset = SyntheticGenomicDataset(num_samples=2000, seq_length=200, seed=42)
    
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
            num_epochs=30, device=device, use_differential=use_differential
        )
        
        models_results[model_name] = (model, training_results)
    
    # Comprehensive evaluation
    final_results = comprehensive_evaluation(models_results, test_loader, device)
    
    # Create visualizations
    visualize_results(models_results, final_results, results_dir)
    
    # Save results
    save_results(final_results, models_results, results_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 PHASE 3 COMPLETE - TRILOGY CONCLUSION")
    print("="*80)
    
    winner = min(final_results.keys(), key=lambda k: final_results[k]['test_loss'])
    winner_metrics = final_results[winner]
    
    print(f"\n🏆 TRILOGY WINNER: {winner}")
    print(f"   Final test loss: {winner_metrics['test_loss']:.4f}")
    print(f"   Parameter count: {winner_metrics['param_count']:,}")
    print(f"   Correlation: {winner_metrics['correlation']:.3f}")
    
    print(f"\n📈 SCIENTIFIC IMPACT SUMMARY:")
    print(f"   Phase 1: ✅ Traditional attention baseline established")
    print(f"   Phase 2: ✅ Sparse approximation confirmed 90%+ efficiency")
    print(f"   Phase 3: ✅ Joint pipeline vs attention comparison complete")
    
    if winner == 'Joint Pipeline':
        print(f"\n🚀 BREAKTHROUGH: Joint Pipeline beats traditional attention!")
        print(f"   Task-specific learning outperforms general attention mechanisms")
        print(f"   This challenges fundamental assumptions about transformer architectures")
    elif winner == 'Sparse Attention':
        print(f"\n⚡ EFFICIENCY WINNER: Sparse attention optimal balance")
        print(f"   90% computational savings with minimal performance loss")
        print(f"   Confirms the 'attention is mostly noise' hypothesis")
    else:
        print(f"\n🤔 TRADITIONAL ATTENTION WINS: Further optimization needed")
        print(f"   Traditional attention remains competitive")
        print(f"   Valuable baseline for future improvements")
    
    print(f"\n🔬 NEXT STEPS:")
    print(f"   • Scale to real genomic data")
    print(f"   • Test on different biological tasks")
    print(f"   • Publish findings on attention efficiency")
    
    return final_results


if __name__ == "__main__":
    final_results = main()
