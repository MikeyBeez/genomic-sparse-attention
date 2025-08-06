"""
Phase 1: Traditional Attention Baseline Training

This script trains a standard multi-head attention model on synthetic genomic data
and extracts attention patterns as ground truth for Phase 2 sparse approximation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Import our custom modules
import sys
sys.path.append('/Users/bard/Code/genomic-sparse-attention/src')

from models.traditional_attention import GenomicAttentionBaseline, AttentionGroundTruthExtractor
from data.synthetic_data import SyntheticGenomicDataGenerator, SyntheticGenomicDataset


class Phase1Trainer:
    """Trainer for Phase 1: Traditional attention baseline."""
    
    def __init__(
        self,
        model: GenomicAttentionBaseline,
        device: str = 'cpu',
        results_dir: str = '/Users/bard/Code/genomic-sparse-attention/results/phase1'
    ):
        self.model = model.to(device)
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(
        self, 
        train_loader: DataLoader, 
        optimizer: torch.optim.Optimizer, 
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions, attention_weights = self.model(sequences)
            predictions = predictions.squeeze()
            
            # Calculate loss
            loss = criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            
            # Binary accuracy (threshold at 0.5)
            predicted_binary = (predictions > 0.5).float()
            actual_binary = (labels > 0.5).float()
            correct_predictions += (predicted_binary == actual_binary).sum().item()
            total_samples += labels.size(0)
            
            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy
    
    def validate_epoch(
        self, 
        val_loader: DataLoader, 
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
                
                # Forward pass
                predictions, _ = self.model(sequences)
                predictions = predictions.squeeze()
                
                # Calculate loss
                loss = criterion(predictions, labels)
                total_loss += loss.item()
                
                # Binary accuracy
                predicted_binary = (predictions > 0.5).float()
                actual_binary = (labels > 0.5).float()
                correct_predictions += (predicted_binary == actual_binary).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy
    
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        patience: int = 10
    ) -> Dict:
        """Full training loop with early stopping."""
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()  # Regression loss for regulatory strength
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        print("=== Phase 1: Training Traditional Attention Baseline ===")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Save best model
                torch.save(self.model.state_dict(), 
                          self.results_dir / 'best_traditional_attention_model.pth')
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break
        
        # Load best model
        self.model.load_state_dict(
            torch.load(self.results_dir / 'best_traditional_attention_model.pth')
        )
        
        # Return training summary
        training_summary = {
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'final_train_accuracy': self.train_accuracies[-1],
            'final_val_accuracy': self.val_accuracies[-1],
            'best_val_loss': best_val_loss,
            'total_epochs': len(self.train_losses),
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }
        
        return training_summary
    
    def visualize_training(self):
        """Create training plots."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss plots
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Loss over Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plots
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Accuracy over Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Binary Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Final metrics
        ax3.bar(['Train Loss', 'Val Loss'], [self.train_losses[-1], self.val_losses[-1]])
        ax3.set_title('Final Loss Comparison')
        ax3.set_ylabel('MSE Loss')
        
        ax4.bar(['Train Acc', 'Val Acc'], [self.train_accuracies[-1], self.val_accuracies[-1]])
        ax4.set_title('Final Accuracy Comparison')
        ax4.set_ylabel('Binary Accuracy')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_attention_patterns(
        self, 
        test_loader: DataLoader, 
        dataset_metadata: List[Dict],
        num_samples: int = 5
    ):
        """Analyze attention patterns for biological interpretability."""
        
        print("\n=== Analyzing Attention Patterns ===")
        
        # Extract ground truth extractor
        extractor = AttentionGroundTruthExtractor(self.model)
        
        # Get a few samples for detailed analysis
        sample_sequences = []
        sample_metadata = []
        
        for i, (sequences, labels) in enumerate(test_loader):
            if len(sample_sequences) >= num_samples:
                break
            sample_sequences.append(sequences[0:1])  # First sequence from batch
            sample_metadata.append(dataset_metadata[i * test_loader.batch_size])
        
        # Visualize attention for these samples
        for i, (seq, meta) in enumerate(zip(sample_sequences, sample_metadata)):
            self._visualize_attention_for_sample(seq, meta, i)
        
        # Create overall attention statistics
        self._create_attention_statistics(test_loader, dataset_metadata)
    
    def _visualize_attention_for_sample(
        self, 
        sequence: torch.Tensor, 
        metadata: Dict, 
        sample_idx: int
    ):
        """Visualize attention pattern for a single sample."""
        
        sequence = sequence.to(self.device)
        attention_patterns = self.model.extract_attention_patterns(sequence)
        
        # Get average attention across heads
        avg_attention = attention_patterns['average_attention'].cpu().numpy()
        
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Plot 1: Attention heatmap
        plt.subplot(2, 1, 1)
        sns.heatmap(avg_attention, cmap='Blues', cbar_kws={'label': 'Attention Weight'})
        plt.title(f'Sample {sample_idx}: Attention Heatmap (Regulatory Strength: {metadata["regulatory_strength"]:.3f})')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        
        # Plot 2: Attention vs. motif positions
        plt.subplot(2, 1, 2)
        attention_sum = avg_attention.sum(axis=0)  # Sum across queries
        positions = range(len(attention_sum))
        
        plt.plot(positions, attention_sum, 'b-', alpha=0.7, label='Attention Sum')
        
        # Mark motif positions
        for motif in metadata['planted_motifs']:
            plt.axvspan(motif['start'], motif['end'], alpha=0.3, color='red', 
                       label=f"{motif['name']} ({motif['strength']:.2f})")
        
        plt.xlabel('Sequence Position')
        plt.ylabel('Total Attention')
        plt.title('Attention vs. Planted Motif Positions')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'attention_sample_{sample_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print motif overlap analysis
        self._analyze_motif_overlap(attention_sum, metadata)
    
    def _analyze_motif_overlap(self, attention_weights: np.ndarray, metadata: Dict):
        """Analyze how well attention overlaps with planted motifs."""
        
        # Create motif mask
        motif_mask = np.zeros_like(attention_weights)
        for motif in metadata['planted_motifs']:
            motif_mask[motif['start']:motif['end']] = 1
        
        # Calculate overlap metrics
        attention_in_motifs = (attention_weights * motif_mask).sum()
        total_attention = attention_weights.sum()
        total_motif_positions = motif_mask.sum()
        
        motif_coverage = attention_in_motifs / total_attention if total_attention > 0 else 0
        attention_precision = attention_in_motifs / total_motif_positions if total_motif_positions > 0 else 0
        
        print(f"\nMotif Overlap Analysis:")
        print(f"  Attention in motif regions: {motif_coverage:.3f} ({attention_in_motifs:.3f}/{total_attention:.3f})")
        print(f"  Motif position coverage: {attention_precision:.3f}")
        print(f"  Total motifs: {len(metadata['planted_motifs'])}")
    
    def _create_attention_statistics(self, test_loader: DataLoader, metadata: List[Dict]):
        """Create overall attention statistics across the test set."""
        
        print("\nCreating attention statistics...")
        
        extractor = AttentionGroundTruthExtractor(self.model)
        ground_truth_data = extractor.extract_ground_truth_dataset(test_loader)
        analysis = extractor.analyze_attention_quality(ground_truth_data)
        
        # Save analysis results
        import json
        with open(self.results_dir / 'attention_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print("Attention analysis saved to attention_analysis.json")
        
        return analysis


def main():
    """Main function to run Phase 1 training."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate synthetic data
    print("=== Generating Synthetic Genomic Data ===")
    generator = SyntheticGenomicDataGenerator(seq_length=2000)
    
    # Training set
    train_sequences, train_labels, train_metadata = generator.generate_dataset(
        num_samples=1000, positive_ratio=0.6, num_motifs_range=(2, 4)
    )
    
    # Validation set
    val_sequences, val_labels, val_metadata = generator.generate_dataset(
        num_samples=200, positive_ratio=0.6, num_motifs_range=(2, 4)
    )
    
    # Test set
    test_sequences, test_labels, test_metadata = generator.generate_dataset(
        num_samples=200, positive_ratio=0.6, num_motifs_range=(2, 4)
    )
    
    # Create datasets and data loaders
    train_dataset = SyntheticGenomicDataset(train_sequences, train_labels, train_metadata)
    val_dataset = SyntheticGenomicDataset(val_sequences, val_labels, val_metadata)
    test_dataset = SyntheticGenomicDataset(test_sequences, test_labels, test_metadata)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Visualize a few samples
    print("\n=== Sample Visualization ===")
    for i in range(3):
        print(generator.visualize_sample(i, train_metadata))
        print("-" * 80)
    
    # Initialize model
    model = GenomicAttentionBaseline(
        vocab_size=6,
        embed_dim=16,
        num_heads=4,
        seq_length=2000,
        dropout=0.1
    )
    
    # Initialize trainer
    trainer = Phase1Trainer(model, device=device)
    
    # Train model
    training_summary = trainer.train_model(
        train_loader, val_loader, 
        num_epochs=50, learning_rate=0.001
    )
    
    print("\n=== Training Summary ===")
    for key, value in training_summary.items():
        print(f"{key}: {value}")
    
    # Visualize training
    trainer.visualize_training()
    
    # Analyze attention patterns
    trainer.analyze_attention_patterns(test_loader, test_metadata, num_samples=3)
    
    # Extract ground truth for Phase 2
    print("\n=== Extracting Ground Truth for Phase 2 ===")
    extractor = AttentionGroundTruthExtractor(model)
    ground_truth_data = extractor.extract_ground_truth_dataset(
        test_loader,
        save_path='/Users/bard/Code/genomic-sparse-attention/data/phase1_ground_truth.pth'
    )
    
    print("Phase 1 complete! Ground truth data saved for Phase 2 sparse approximation.")
    print(f"Results saved in: {trainer.results_dir}")


if __name__ == "__main__":
    main()
