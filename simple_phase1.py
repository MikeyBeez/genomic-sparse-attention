#!/usr/bin/env python3
"""
Simplified Phase 1: Traditional Attention Baseline

This is a working version focused on the core validation of your hypothesis.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import os
from pathlib import Path

print("🧬 Phase 1: Traditional Attention Baseline (Simplified)")
print("=" * 60)

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Create results directory
results_dir = Path('/Users/bard/Code/genomic-sparse-attention/results/phase1')
results_dir.mkdir(parents=True, exist_ok=True)

class SimpleGenomicDataset(Dataset):
    """Simplified genomic dataset with planted motifs."""
    
    def __init__(self, num_samples=1000, seq_length=200):
        self.seq_length = seq_length
        self.char_to_idx = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4, 'PAD': 5}
        
        # Define regulatory motifs
        self.motifs = [
            {"name": "TATA", "seq": "TATAAA", "strength": 0.8},
            {"name": "CAAT", "seq": "CCAAT", "strength": 0.6},
            {"name": "GC", "seq": "GGGCGG", "strength": 0.7}
        ]
        
        print(f"Generating {num_samples} synthetic sequences...")
        self.sequences, self.labels, self.metadata = self._generate_dataset(num_samples)
        print(f"✅ Generated dataset with mean regulatory strength: {self.labels.mean():.3f}")
    
    def _generate_dataset(self, num_samples):
        sequences = []
        labels = []
        metadata = []
        
        for i in range(num_samples):
            sequence, motifs_planted, strength = self._generate_single_sequence()
            
            # Encode sequence
            encoded = torch.tensor([self.char_to_idx.get(c, 4) for c in sequence], dtype=torch.long)
            sequences.append(encoded)
            labels.append(strength)
            metadata.append({
                'sequence': sequence,
                'planted_motifs': motifs_planted,
                'strength': strength
            })
            
            if (i + 1) % 200 == 0:
                print(f"  Generated {i + 1}/{num_samples} sequences...")
        
        return torch.stack(sequences), torch.tensor(labels, dtype=torch.float32), metadata
    
    def _generate_single_sequence(self):
        """Generate single sequence with planted motifs."""
        # Random background sequence
        nucleotides = ['A', 'T', 'C', 'G']
        sequence = [random.choice(nucleotides) for _ in range(self.seq_length)]
        
        # Plant 1-3 motifs
        num_motifs = random.randint(1, min(3, len(self.motifs)))
        selected_motifs = random.sample(self.motifs, num_motifs)
        
        planted_motifs = []
        total_strength = 0
        
        for motif in selected_motifs:
            # Find valid position (avoid overlaps)
            max_pos = self.seq_length - len(motif["seq"])
            if max_pos > 10:
                position = random.randint(10, max_pos - 10)
                
                # Check for overlaps
                overlap = False
                for existing in planted_motifs:
                    if abs(position - existing['start']) < 10:
                        overlap = True
                        break
                
                if not overlap:
                    # Plant motif
                    for j, nucleotide in enumerate(motif["seq"]):
                        sequence[position + j] = nucleotide
                    
                    planted_motifs.append({
                        'name': motif['name'],
                        'start': position,
                        'end': position + len(motif["seq"]),
                        'sequence': motif["seq"],
                        'strength': motif['strength']
                    })
                    total_strength += motif['strength']
        
        # Calculate regulatory strength
        regulatory_strength = min(1.0, total_strength / max(1, len(planted_motifs)))
        
        return ''.join(sequence), planted_motifs, regulatory_strength
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class SimpleGenomicAttention(nn.Module):
    """Simplified genomic attention model."""
    
    def __init__(self, vocab_size=6, embed_dim=32, seq_length=200):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        
        # Embeddings
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=5)
        
        # Simple self-attention (single head for simplicity)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=2, batch_first=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Embed sequences
        embedded = self.embeddings(x)  # [batch, seq_len, embed_dim]
        
        # Self-attention
        attended, attention_weights = self.attention(embedded, embedded, embedded)
        
        # Global average pooling
        pooled = attended.mean(dim=1)  # [batch, embed_dim]
        
        # Predict regulatory strength
        output = self.classifier(pooled).squeeze(-1)  # [batch]
        
        return output, attention_weights


def train_model(model, train_loader, val_loader, num_epochs=20, device='cpu'):
    """Train the model and return training history."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    print(f"\n🏋️ Training model for {num_epochs} epochs...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions, _ = model(sequences)
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
                predictions, _ = model(sequences)
                val_loss += criterion(predictions, labels).item()
        
        val_losses.append(val_loss / len(val_loader))
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}")
    
    return train_losses, val_losses


def analyze_attention(model, dataset, device='cpu', num_samples=3):
    """Analyze attention patterns vs planted motifs."""
    
    print(f"\n🔍 Analyzing attention patterns on {num_samples} samples...")
    model.eval()
    
    attention_overlaps = []
    
    for i in range(num_samples):
        sequence = dataset.sequences[i:i+1].to(device)
        metadata = dataset.metadata[i]
        
        with torch.no_grad():
            _, attention_weights = model(sequence)
        
        # Average attention across heads
        avg_attention = attention_weights.mean(dim=1).cpu().numpy()[0]  # [seq_len, seq_len]
        
        # Sum attention received by each position
        attention_focus = avg_attention.sum(axis=0)
        
        # Create motif mask
        seq_len = dataset.seq_length
        motif_mask = np.zeros(seq_len)
        for motif in metadata['planted_motifs']:
            motif_mask[motif['start']:motif['end']] = 1
        
        # Calculate overlap
        total_attention = attention_focus.sum()
        attention_in_motifs = (attention_focus * motif_mask).sum()
        overlap_ratio = attention_in_motifs / total_attention if total_attention > 0 else 0
        
        attention_overlaps.append(overlap_ratio)
        
        print(f"Sample {i+1}:")
        print(f"  Planted motifs: {[m['name'] for m in metadata['planted_motifs']]}")
        print(f"  Regulatory strength: {metadata['strength']:.3f}")
        print(f"  Attention overlap with motifs: {overlap_ratio:.3f}")
        
        # Show which positions got highest attention
        top_positions = np.argsort(attention_focus)[-10:]
        print(f"  Top attention positions: {top_positions.tolist()}")
        print()
    
    avg_overlap = np.mean(attention_overlaps)
    print(f"📊 Average attention overlap with planted motifs: {avg_overlap:.3f}")
    print(f"Random baseline would be: ~{np.mean([len(m['planted_motifs']) * 6 / dataset.seq_length for m in dataset.metadata[:num_samples]]):.3f}")
    
    return attention_overlaps


def main():
    """Main experiment function."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate data
    print("\n📊 Generating synthetic genomic data...")
    train_dataset = SimpleGenomicDataset(num_samples=800, seq_length=200)
    val_dataset = SimpleGenomicDataset(num_samples=200, seq_length=200)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Show some examples
    print("\n📋 Sample sequences:")
    for i in range(3):
        meta = train_dataset.metadata[i]
        print(f"Sample {i+1}: {len(meta['planted_motifs'])} motifs, strength {meta['strength']:.3f}")
        for motif in meta['planted_motifs']:
            print(f"  {motif['name']} at positions {motif['start']}-{motif['end']}")
    
    # Create and train model
    print("\n🤖 Creating attention model...")
    model = SimpleGenomicAttention(vocab_size=6, embed_dim=32, seq_length=200)
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=25, device=device)
    
    # Save model
    model_path = results_dir / 'simple_attention_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Analyze attention patterns
    attention_overlaps = analyze_attention(model, val_dataset, device=device, num_samples=5)
    
    # Summary
    print("\n" + "="*60)
    print("🎯 PHASE 1 RESULTS SUMMARY")
    print("="*60)
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Average attention-motif overlap: {np.mean(attention_overlaps):.3f}")
    
    if np.mean(attention_overlaps) > 0.2:
        print("✅ SUCCESS: Attention shows some focus on regulatory motifs!")
        print("   Ready for Phase 2: Sparse attention approximation")
    else:
        print("⚠️  Attention overlap is low - may need tuning")
    
    print(f"\nResults saved in: {results_dir}")
    
    # Create ground truth data for Phase 2
    print("\n💾 Extracting ground truth for Phase 2...")
    ground_truth_data = {
        'model_state': model.state_dict(),
        'train_dataset_sequences': train_dataset.sequences,
        'train_dataset_labels': train_dataset.labels,
        'val_dataset_sequences': val_dataset.sequences,
        'val_dataset_labels': val_dataset.labels,
        'attention_overlaps': attention_overlaps,
        'model_params': sum(p.numel() for p in model.parameters())
    }
    
    ground_truth_path = results_dir.parent / 'phase1_ground_truth_simple.pth'
    torch.save(ground_truth_data, ground_truth_path)
    print(f"✅ Ground truth saved to {ground_truth_path}")
    
    return model, train_dataset, val_dataset, attention_overlaps


if __name__ == "__main__":
    model, train_dataset, val_dataset, overlaps = main()
