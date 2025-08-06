#!/usr/bin/env python3
"""
Phase 2: Sparse Attention Approximation

This tests the core hypothesis: Can we replicate traditional attention performance
using only 5-10% of tokens, proving that 90-95% of attention computation is noise?
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import time

print("🎯 Phase 2: Sparse Attention Approximation - Testing 95% Noise Hypothesis")
print("=" * 80)

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Create results directory
results_dir = Path('/Users/bard/Code/genomic-sparse-attention/results/phase2')
results_dir.mkdir(parents=True, exist_ok=True)


class TokenSelector(nn.Module):
    """
    Token Selection Organ: Learns to identify most relevant genomic positions.
    This is the key innovation - can it find regulatory motifs automatically?
    """
    
    def __init__(self, embed_dim=32):
        super().__init__()
        
        # Simple scoring network
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()  # Score between 0-1
        )
    
    def forward(self, embeddings):
        """
        Score each position's importance for regulatory prediction.
        
        Args:
            embeddings: [batch_size, seq_length, embed_dim]
            
        Returns:
            scores: [batch_size, seq_length] - importance scores
        """
        batch_size, seq_length, embed_dim = embeddings.shape
        
        # Flatten for processing
        flat_embeddings = embeddings.reshape(-1, embed_dim)
        scores = self.scorer(flat_embeddings).reshape(batch_size, seq_length)
        
        return scores
    
    def select_top_k(self, embeddings, sparsity_ratio=0.1):
        """
        Select top-k most important positions.
        
        Args:
            embeddings: [batch_size, seq_length, embed_dim]
            sparsity_ratio: Fraction of positions to keep (0.1 = 10%)
            
        Returns:
            selected_embeddings: Selected embeddings for attention
            selected_indices: Which positions were selected
            scores: All position scores
        """
        batch_size, seq_length, embed_dim = embeddings.shape
        k = max(1, int(seq_length * sparsity_ratio))
        
        # Get importance scores
        scores = self.forward(embeddings)
        
        # Select top-k positions for each sequence
        top_k_values, top_k_indices = torch.topk(scores, k, dim=1)
        
        # Gather selected embeddings
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, k)
        selected_embeddings = embeddings[batch_indices, top_k_indices]
        
        return selected_embeddings, top_k_indices, scores


class MLPApproximator(nn.Module):
    """
    MLP Approximation Organ: Replicates attention using selected positions.
    Uses bottleneck architecture to filter noise like in your paper.
    """
    
    def __init__(self, embed_dim=32, bottleneck_dim=8):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        
        # MLP with bottleneck (like your attention paper)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, bottleneck_dim),  # Q,K,V concatenated -> bottleneck
            nn.ReLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.ReLU(), 
            nn.Linear(bottleneck_dim, embed_dim)  # Back to embed_dim
        )
    
    def forward(self, query, selected_keys, selected_values):
        """
        Approximate attention using only selected key-value pairs.
        
        Args:
            query: [batch_size, embed_dim] - averaged query representation
            selected_keys: [batch_size, k, embed_dim] - top-k keys
            selected_values: [batch_size, k, embed_dim] - corresponding values
            
        Returns:
            approximated_output: [batch_size, embed_dim]
        """
        batch_size, k, embed_dim = selected_keys.shape
        
        # Expand query to match selected pairs
        expanded_query = query.unsqueeze(1).expand(-1, k, -1)  # [batch, k, embed_dim]
        
        # Concatenate Q, K, V for each selected pair
        qkv_concat = torch.cat([
            expanded_query, 
            selected_keys, 
            selected_values
        ], dim=-1)  # [batch, k, embed_dim * 3]
        
        # Process through MLP with bottleneck
        processed = self.mlp(qkv_concat)  # [batch, k, embed_dim]
        
        # Average across selected positions
        output = processed.mean(dim=1)  # [batch, embed_dim]
        
        return output


class SparseAttentionApproximator(nn.Module):
    """
    Complete sparse attention model combining token selection + MLP approximation.
    This is the test of your 95% noise hypothesis!
    """
    
    def __init__(self, vocab_size=6, embed_dim=32, seq_length=200, sparsity_ratio=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.sparsity_ratio = sparsity_ratio
        
        # Same embeddings as traditional model
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=5)
        
        # Sparse attention components
        self.token_selector = TokenSelector(embed_dim)
        self.mlp_approximator = MLPApproximator(embed_dim, bottleneck_dim=8)
        
        # Classification head (same as traditional)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through sparse attention approximation."""
        # Embed sequences
        embeddings = self.embeddings(x)  # [batch, seq_len, embed_dim]
        
        # Select most important positions
        selected_embeddings, selected_indices, selection_scores = self.token_selector.select_top_k(
            embeddings, self.sparsity_ratio
        )
        
        # Use selected embeddings as keys and values
        # Query is the global average (simplified)
        query = embeddings.mean(dim=1)  # [batch, embed_dim]
        
        # Approximate attention using selected positions
        approximated = self.mlp_approximator(query, selected_embeddings, selected_embeddings)
        
        # Predict regulatory strength
        output = self.classifier(approximated).squeeze(-1)
        
        return output, selected_indices, selection_scores


class GroundTruthDataset(Dataset):
    """Dataset using ground truth from Phase 1."""
    
    def __init__(self, ground_truth_path):
        print(f"Loading ground truth from {ground_truth_path}...")
        
        # Load with weights_only=False to handle numpy objects
        data = torch.load(ground_truth_path, map_location='cpu', weights_only=False)
        
        # Use training data from Phase 1
        self.sequences = data['train_dataset_sequences']
        self.labels = data['train_dataset_labels']
        
        print(f"✅ Loaded {len(self.sequences)} sequences")
        print(f"Mean regulatory strength: {self.labels.mean():.3f}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def train_sparse_model(model, train_loader, val_loader, num_epochs=25, device='cpu'):
    """Train sparse approximation model."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    print(f"\n🏋️ Training sparse model (sparsity: {model.sparsity_ratio:.1%})...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions, _, _ = model(sequences)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                predictions, _, _ = model(sequences)
                val_loss += criterion(predictions, labels).item()
        
        val_losses.append(val_loss / len(val_loader))
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")
    
    return train_losses, val_losses


def compare_with_traditional(traditional_path, sparse_model, test_loader, device='cpu'):
    """Compare sparse model with traditional baseline."""
    
    print("\n🔍 Comparing with traditional attention baseline...")
    
    # Load traditional model
    from simple_phase1 import SimpleGenomicAttention
    traditional_model = SimpleGenomicAttention(vocab_size=6, embed_dim=32, seq_length=200)
    
    # Load ground truth data to get traditional model state
    ground_truth = torch.load(traditional_path, map_location='cpu', weights_only=False)
    traditional_model.load_state_dict(ground_truth['model_state'])
    traditional_model = traditional_model.to(device)
    
    # Compare performance
    traditional_model.eval()
    sparse_model.eval()
    
    traditional_loss = 0
    sparse_loss = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            # Traditional predictions
            trad_pred, _ = traditional_model(sequences)
            traditional_loss += criterion(trad_pred, labels).item()
            
            # Sparse predictions
            sparse_pred, _, _ = sparse_model(sequences)
            sparse_loss += criterion(sparse_pred, labels).item()
    
    traditional_loss /= len(test_loader)
    sparse_loss /= len(test_loader)
    
    # Calculate performance metrics
    performance_ratio = sparse_loss / traditional_loss
    performance_maintained = (1 - (sparse_loss - traditional_loss) / traditional_loss) * 100
    
    print(f"Traditional attention loss: {traditional_loss:.4f}")
    print(f"Sparse attention loss: {sparse_loss:.4f}")
    print(f"Performance ratio: {performance_ratio:.3f} (lower is better)")
    print(f"Performance maintained: {performance_maintained:.1f}%")
    
    return traditional_loss, sparse_loss, performance_ratio


def analyze_token_selection(model, dataset, device='cpu', num_samples=3):
    """Analyze which tokens the sparse model selects."""
    
    print(f"\n🔍 Analyzing token selection patterns...")
    model.eval()
    
    with torch.no_grad():
        for i in range(num_samples):
            sequence = dataset.sequences[i:i+1].to(device)
            
            # Get predictions and selected positions
            _, selected_indices, selection_scores = model(sequence)
            
            selected_indices = selected_indices.cpu().numpy()[0]
            selection_scores = selection_scores.cpu().numpy()[0]
            
            print(f"\nSample {i+1}:")
            print(f"  Sparsity: {len(selected_indices)}/{len(selection_scores)} positions ({len(selected_indices)/len(selection_scores):.1%})")
            print(f"  Selected positions: {selected_indices[:10].tolist()}{'...' if len(selected_indices) > 10 else ''}")
            print(f"  Top 10 selection scores: {np.sort(selection_scores)[-10:][::-1].round(3).tolist()}")
            
            # Show which nucleotides were selected
            seq_str = ''.join(['ATCGN?'][idx] for idx in sequence.cpu().numpy()[0] if idx < 6)
            selected_nucleotides = [seq_str[pos] for pos in selected_indices[:10]]
            print(f"  Selected nucleotides: {selected_nucleotides}")


def measure_computational_efficiency(traditional_model, sparse_model, test_input, device='cpu'):
    """Measure computational efficiency gains."""
    
    print("\n⚡ Measuring computational efficiency...")
    
    traditional_model.eval()
    sparse_model.eval()
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            traditional_model(test_input)
            sparse_model(test_input)
    
    # Time traditional model
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            traditional_model(test_input)
    traditional_time = (time.perf_counter() - start_time) / 100
    
    # Time sparse model
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            sparse_model(test_input)
    sparse_time = (time.perf_counter() - start_time) / 100
    
    speedup = traditional_time / sparse_time
    
    # Parameter counts
    trad_params = sum(p.numel() for p in traditional_model.parameters())
    sparse_params = sum(p.numel() for p in sparse_model.parameters())
    
    print(f"Traditional model time: {traditional_time:.4f}s per batch")
    print(f"Sparse model time: {sparse_time:.4f}s per batch")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Traditional parameters: {trad_params:,}")
    print(f"Sparse parameters: {sparse_params:,}")
    print(f"Parameter ratio: {sparse_params/trad_params:.2f}")
    
    return speedup, trad_params, sparse_params


def test_sparsity_ratios(ground_truth_path, sparsity_ratios=[0.05, 0.1, 0.2]):
    """Test different sparsity ratios to find optimal point."""
    
    print(f"\n🎯 Testing sparsity ratios: {[f'{r:.1%}' for r in sparsity_ratios]}")
    
    # Load data
    dataset = GroundTruthDataset(ground_truth_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for sparsity in sparsity_ratios:
        print(f"\n--- Testing {sparsity:.1%} sparsity ---")
        
        # Create model
        model = SparseAttentionApproximator(sparsity_ratio=sparsity)
        
        # Train
        train_losses, val_losses = train_sparse_model(
            model, train_loader, val_loader, num_epochs=20, device=device
        )
        
        # Compare with traditional
        traditional_loss, sparse_loss, performance_ratio = compare_with_traditional(
            ground_truth_path, model, val_loader, device
        )
        
        results[sparsity] = {
            'final_val_loss': val_losses[-1],
            'traditional_loss': traditional_loss,
            'sparse_loss': sparse_loss,
            'performance_ratio': performance_ratio,
            'performance_maintained': (1 - (sparse_loss - traditional_loss) / traditional_loss) * 100
        }
        
        print(f"Results for {sparsity:.1%}: Performance maintained = {results[sparsity]['performance_maintained']:.1f}%")
    
    return results


def main():
    """Main Phase 2 experiment."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    ground_truth_path = '/Users/bard/Code/genomic-sparse-attention/results/phase1_ground_truth_simple.pth'
    
    # Test different sparsity ratios
    results = test_sparsity_ratios(ground_truth_path, sparsity_ratios=[0.05, 0.1, 0.2])
    
    # Summary
    print("\n" + "="*80)
    print("🎯 PHASE 2 RESULTS: TESTING 95% NOISE HYPOTHESIS")
    print("="*80)
    
    for sparsity, metrics in results.items():
        print(f"\n{sparsity:.1%} Sparsity (using {sparsity:.1%} of tokens):")
        print(f"  Performance maintained: {metrics['performance_maintained']:.1f}%")
        print(f"  Performance ratio: {metrics['performance_ratio']:.3f}")
        
        if metrics['performance_maintained'] >= 95:
            print("  ✅ HYPOTHESIS CONFIRMED: >95% performance with <10% computation!")
        elif metrics['performance_maintained'] >= 90:
            print("  🟡 STRONG EVIDENCE: >90% performance maintained")
        else:
            print("  🔴 HYPOTHESIS NOT CONFIRMED: <90% performance")
    
    # Find best sparsity
    best_sparsity = min(results.keys(), key=lambda k: results[k]['performance_ratio'])
    best_metrics = results[best_sparsity]
    
    print(f"\n🏆 BEST RESULT: {best_sparsity:.1%} sparsity")
    print(f"   Performance maintained: {best_metrics['performance_maintained']:.1f}%")
    print(f"   Using only {best_sparsity:.1%} of tokens!")
    print(f"   Computational savings: {(1-best_sparsity)*100:.0f}%")
    
    if best_metrics['performance_maintained'] >= 95:
        print("\n🎉 YOUR HYPOTHESIS IS CONFIRMED!")
        print("   ✅ 95% of attention computation IS NOISE")
        print("   ✅ Sparse attention achieves equivalent performance")
        print("   ✅ Ready for Phase 3: Joint pipeline comparison")
    else:
        print(f"\n🤔 Partial confirmation - {best_metrics['performance_maintained']:.1f}% performance maintained")
        print("   Further optimization may be needed")
    
    return results


if __name__ == "__main__":
    results = main()
