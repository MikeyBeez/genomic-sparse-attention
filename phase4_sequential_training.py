#!/usr/bin/env python3
"""
Phase 4: Sequential Training Pipeline - The Ultimate Test

This tests the ultimate hypothesis: Can joint pipeline training TEACH attention mechanisms
what to focus on, leading to superior sparse attention?

Sequential Process:
1. Train joint pipeline first (learn task-specific representations)
2. Extract learned embeddings and attention patterns  
3. Initialize sparse attention with joint pipeline knowledge
4. Fine-tune sparse attention for final optimization

This tests whether "teaching" sparse attention leads to better results than training from scratch.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import json
import pickle

print("🎯 Phase 4: Sequential Training Pipeline - The Ultimate Test")
print("=" * 80)
print("Question: Can joint pipeline training TEACH sparse attention what to focus on?")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create results directory
results_dir = Path('/Users/bard/Code/genomic-sparse-attention/results/phase4')
results_dir.mkdir(parents=True, exist_ok=True)


class TeacherJointPipeline(nn.Module):
    """
    Teacher Model: Joint pipeline that learns task-specific representations.
    This will teach the sparse attention model what patterns matter.
    """
    
    def __init__(self, vocab_size=6, tiny_embed_dim=3, seq_length=200):
        super().__init__()
        
        self.tiny_embed_dim = tiny_embed_dim
        self.seq_length = seq_length
        
        print(f"🧠 Teacher Model Setup:")
        print(f"   Embedding size: {tiny_embed_dim}D per token")
        print(f"   Will learn task-specific patterns to teach student")
        
        # Task-specific embeddings
        self.embeddings = nn.Embedding(vocab_size, tiny_embed_dim, padding_idx=5)
        
        # Multi-scale pattern detectors (this learns what to focus on)
        self.local_detector = nn.Conv1d(tiny_embed_dim, 16, kernel_size=3, padding=1)
        self.motif_detector = nn.Conv1d(tiny_embed_dim, 16, kernel_size=7, padding=3)  
        self.long_detector = nn.Conv1d(tiny_embed_dim, 16, kernel_size=15, padding=7)
        
        # Pattern integration
        self.integrator = nn.Sequential(
            nn.Conv1d(48, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Final predictor
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
        nn.init.xavier_uniform_(self.embeddings.weight)
        
    def forward(self, x, return_patterns=False):
        """Forward pass with optional pattern extraction for teaching."""
        # Get embeddings
        embeddings = self.embeddings(x)  # [batch, seq, embed_dim]
        embeddings_t = embeddings.transpose(1, 2)  # [batch, embed_dim, seq]
        
        # Multi-scale pattern detection
        local_patterns = torch.relu(self.local_detector(embeddings_t))
        motif_patterns = torch.relu(self.motif_detector(embeddings_t))
        long_patterns = torch.relu(self.long_detector(embeddings_t))
        
        # Combine patterns
        combined = torch.cat([local_patterns, motif_patterns, long_patterns], dim=1)
        integrated = self.integrator(combined).squeeze(-1)
        
        # Make prediction
        prediction = self.predictor(integrated).squeeze(-1)
        
        if return_patterns:
            # Return both prediction and learned patterns for teaching
            pattern_importance = {
                'local': local_patterns.mean(dim=1),  # [batch, seq]
                'motif': motif_patterns.mean(dim=1),  # [batch, seq] 
                'long': long_patterns.mean(dim=1),    # [batch, seq]
                'embeddings': embeddings  # [batch, seq, embed_dim]
            }
            return prediction, pattern_importance
        
        return prediction
    
    def extract_attention_guidance(self, dataloader, device='cpu'):
        """Extract learned patterns to guide sparse attention."""
        self.eval()
        
        attention_guidance = {
            'position_importance': [],
            'pattern_weights': [],
            'embedding_prototypes': []
        }
        
        print("\n🔍 Extracting learned patterns from teacher model...")
        
        with torch.no_grad():
            for sequences, labels in dataloader:
                sequences = sequences.to(device)
                
                prediction, patterns = self.forward(sequences, return_patterns=True)
                
                # Calculate position importance (which positions the teacher focuses on)
                combined_importance = (
                    patterns['local'] + 
                    patterns['motif'] + 
                    patterns['long']
                ) / 3.0  # [batch, seq]
                
                attention_guidance['position_importance'].append(combined_importance.cpu())
                
                # Extract successful pattern examples (high regulatory strength predictions)
                high_reg_mask = prediction > 0.7
                if high_reg_mask.sum() > 0:
                    important_embeddings = patterns['embeddings'][high_reg_mask]
                    attention_guidance['embedding_prototypes'].append(important_embeddings.cpu())
        
        # Aggregate guidance
        all_importance = torch.cat(attention_guidance['position_importance'], dim=0)
        position_importance_mean = all_importance.mean(dim=0)  # [seq_length]
        
        if attention_guidance['embedding_prototypes']:
            all_prototypes = torch.cat(attention_guidance['embedding_prototypes'], dim=0)
            prototype_mean = all_prototypes.mean(dim=0)  # [seq_length, embed_dim]
        else:
            prototype_mean = None
        
        guidance = {
            'position_importance': position_importance_mean,
            'embedding_prototypes': prototype_mean,
            'top_positions': torch.topk(position_importance_mean, k=20)[1].tolist()
        }
        
        print(f"✅ Extracted guidance:")
        print(f"   Top important positions: {guidance['top_positions'][:10]}")
        print(f"   Mean position importance: {position_importance_mean.mean():.4f}")
        
        return guidance


class GuidedSparseAttention(nn.Module):
    """
    Student Model: Sparse attention initialized with teacher's knowledge.
    Uses guidance from joint pipeline to focus on the right positions.
    """
    
    def __init__(self, vocab_size=6, embed_dim=32, seq_length=200, sparsity_ratio=0.1, guidance=None):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.sparsity_ratio = sparsity_ratio
        self.guidance = guidance
        
        print(f"🎓 Student Model Setup:")
        print(f"   Embedding size: {embed_dim}D per token")
        print(f"   Sparsity ratio: {sparsity_ratio:.1%}")
        print(f"   Using teacher guidance: {'Yes' if guidance else 'No'}")
        
        # Embeddings (larger than teacher for capacity)
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=5)
        
        # Guided token selector (uses teacher's wisdom)
        self.token_selector = GuidedTokenSelector(embed_dim, guidance)
        
        # MLP approximator for attention
        self.mlp_approximator = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize with teacher guidance if available
        if guidance and guidance['embedding_prototypes'] is not None:
            self._initialize_with_guidance(guidance)
    
    def _initialize_with_guidance(self, guidance):
        """Initialize embeddings using teacher's learned patterns."""
        print("🔗 Initializing student with teacher guidance...")
        
        # The teacher uses smaller embeddings, so we need to expand them
        teacher_prototypes = guidance['embedding_prototypes']  # [seq, teacher_embed_dim]
        teacher_embed_dim = teacher_prototypes.shape[1]
        
        # Expand teacher embeddings to student size
        with torch.no_grad():
            for i in range(6):  # vocab_size
                if i < teacher_prototypes.shape[0]:
                    # Use teacher's pattern, expanded with random padding
                    teacher_pattern = teacher_prototypes[i % teacher_prototypes.shape[0]]
                    expanded = torch.cat([
                        teacher_pattern,
                        torch.randn(self.embed_dim - teacher_embed_dim) * 0.01
                    ])
                    self.embeddings.weight[i] = expanded
    
    def forward(self, x):
        """Forward pass with guided sparse attention."""
        embeddings = self.embeddings(x)  # [batch, seq, embed_dim]
        
        # Use guided token selection
        selected_embeddings, selected_indices, importance_scores = self.token_selector.select_guided_tokens(
            embeddings, self.sparsity_ratio
        )
        
        # Approximate attention using selected tokens
        batch_size, k, embed_dim = selected_embeddings.shape
        query = embeddings.mean(dim=1).unsqueeze(1).expand(-1, k, -1)
        
        # Q, K, V approximation
        qkv_concat = torch.cat([query, selected_embeddings, selected_embeddings], dim=-1)
        processed = self.mlp_approximator(qkv_concat.reshape(-1, self.embed_dim * 3))
        processed = processed.reshape(batch_size, k, self.embed_dim)
        
        # Pool and classify
        pooled = processed.mean(dim=1)
        prediction = self.classifier(pooled).squeeze(-1)
        
        return prediction, selected_indices, importance_scores


class GuidedTokenSelector(nn.Module):
    """
    Token selector that uses guidance from the teacher model.
    Combines learned position importance with trainable selection.
    """
    
    def __init__(self, embed_dim, guidance=None):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.guidance = guidance
        
        # Trainable token importance scorer
        self.importance_scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Guidance weights (how much to trust teacher vs learn from scratch)
        if guidance:
            self.register_buffer('position_prior', guidance['position_importance'])
            self.guidance_weight = nn.Parameter(torch.tensor(0.3))  # Start trusting teacher 30%
            print(f"   📚 Position prior loaded: {len(guidance['position_importance'])} positions")
        else:
            self.position_prior = None
            self.guidance_weight = None
    
    def select_guided_tokens(self, embeddings, sparsity_ratio):
        """Select tokens using both learned importance and teacher guidance."""
        batch_size, seq_length, embed_dim = embeddings.shape
        k = max(1, int(seq_length * sparsity_ratio))
        
        # Get trainable importance scores
        flat_embeddings = embeddings.reshape(-1, embed_dim)
        learned_scores = self.importance_scorer(flat_embeddings).reshape(batch_size, seq_length)
        
        if self.guidance and self.position_prior is not None:
            # Combine learned scores with teacher guidance
            position_prior = self.position_prior.unsqueeze(0).expand(batch_size, -1).to(embeddings.device)
            
            # Weighted combination: guidance_weight * prior + (1 - guidance_weight) * learned
            combined_scores = (
                self.guidance_weight * position_prior + 
                (1 - self.guidance_weight) * learned_scores
            )
            
            final_scores = torch.sigmoid(combined_scores)
        else:
            final_scores = learned_scores
        
        # Select top-k positions
        top_k_values, top_k_indices = torch.topk(final_scores, k, dim=1)
        
        # Gather selected embeddings
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, k)
        selected_embeddings = embeddings[batch_indices, top_k_indices]
        
        return selected_embeddings, top_k_indices, final_scores


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


def train_teacher_model(teacher, train_loader, val_loader, num_epochs=30, device='cpu'):
    """Train the teacher joint pipeline model."""
    
    print("\n" + "="*60)
    print("👨‍🏫 STAGE 1: TRAINING TEACHER MODEL")
    print("="*60)
    
    teacher = teacher.to(device)
    optimizer = optim.Adam(teacher.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"🏋️ Training teacher model...")
    print(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        teacher.train()
        train_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = teacher(sequences)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        teacher.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                predictions = teacher(sequences)
                val_loss += criterion(predictions, labels).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}: Train = {train_loss:.4f}, Val = {val_loss:.4f}")
    
    print(f"✅ Teacher training complete! Best validation loss: {best_val_loss:.4f}")
    return {'best_val_loss': best_val_loss, 'final_train_loss': train_loss}


def train_student_model(student, train_loader, val_loader, num_epochs=25, device='cpu'):
    """Train the student sparse attention model with teacher guidance."""
    
    print("\n" + "="*60)
    print("🎓 STAGE 2: TRAINING STUDENT MODEL (with teacher guidance)")
    print("="*60)
    
    student = student.to(device)
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"🏋️ Training guided student model...")
    print(f"Student parameters: {sum(p.numel() for p in student.parameters()):,}")
    
    if hasattr(student.token_selector, 'guidance_weight') and student.token_selector.guidance_weight is not None:
        print(f"📚 Initial guidance weight: {student.token_selector.guidance_weight.item():.3f}")
    
    for epoch in range(num_epochs):
        # Training
        student.train()
        train_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions, _, _ = student(sequences)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation  
        student.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                predictions, _, _ = student(sequences)
                val_loss += criterion(predictions, labels).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 5 == 0:
            guidance_weight = "N/A"
            if hasattr(student.token_selector, 'guidance_weight') and student.token_selector.guidance_weight is not None:
                guidance_weight = f"{student.token_selector.guidance_weight.item():.3f}"
            
            print(f"Epoch {epoch+1:2d}: Train = {train_loss:.4f}, Val = {val_loss:.4f}, Guidance = {guidance_weight}")
    
    print(f"✅ Student training complete!")
    return {'final_train_loss': train_loss, 'final_val_loss': val_loss}


def evaluate_models(models_dict, test_loader, device='cpu'):
    """Evaluate all models on test set."""
    
    print("\n" + "="*80)
    print("🎯 STAGE 3: FINAL EVALUATION - Teacher vs Student Comparison")
    print("="*80)
    
    criterion = nn.MSELoss()
    results = {}
    
    for model_name, model in models_dict.items():
        model.eval()
        test_loss = 0
        predictions_list = []
        labels_list = []
        
        print(f"\n📊 Evaluating {model_name}...")
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                
                if 'Teacher' in model_name:
                    predictions = model(sequences)
                else:  # Student model
                    predictions, _, _ = model(sequences)
                
                test_loss += criterion(predictions, labels).item()
                predictions_list.extend(predictions.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
        
        test_loss /= len(test_loader)
        predictions_array = np.array(predictions_list)
        labels_array = np.array(labels_list)
        
        mae = np.mean(np.abs(predictions_array - labels_array))
        correlation = np.corrcoef(predictions_array, labels_array)[0, 1]
        param_count = sum(p.numel() for p in model.parameters())
        
        results[model_name] = {
            'test_loss': float(test_loss),
            'mae': float(mae), 
            'correlation': float(correlation),
            'param_count': int(param_count)
        }
        
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   Correlation: {correlation:.3f}")
        print(f"   Parameters: {param_count:,}")
    
    return results


def main():
    """Main sequential training experiment."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate dataset
    print("\n🧬 Generating synthetic genomic dataset...")
    full_dataset = SyntheticGenomicDataset(num_samples=1800, seq_length=200, seed=42)
    
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
    
    # STAGE 1: Train teacher model
    teacher_model = TeacherJointPipeline(tiny_embed_dim=3)
    teacher_results = train_teacher_model(teacher_model, train_loader, val_loader, 
                                        num_epochs=30, device=device)
    
    # Extract guidance from trained teacher
    guidance = teacher_model.extract_attention_guidance(train_loader, device)
    
    # Save guidance for analysis
    with open(results_dir / 'teacher_guidance.pkl', 'wb') as f:
        pickle.dump(guidance, f)
    
    # STAGE 2: Train student models (with and without guidance)
    student_guided = GuidedSparseAttention(
        embed_dim=32, sparsity_ratio=0.1, guidance=guidance
    )
    
    student_baseline = GuidedSparseAttention(
        embed_dim=32, sparsity_ratio=0.1, guidance=None  # No teacher guidance
    )
    
    print(f"\n📚 Training student models...")
    student_guided_results = train_student_model(student_guided, train_loader, val_loader, 
                                                num_epochs=25, device=device)
    
    print(f"\n🔄 Training baseline student (no guidance)...")
    student_baseline_results = train_student_model(student_baseline, train_loader, val_loader,
                                                 num_epochs=25, device=device)
    
    # STAGE 3: Final evaluation
    models = {
        'Teacher (Joint Pipeline)': teacher_model,
        'Student (Guided Sparse)': student_guided,
        'Student (Baseline Sparse)': student_baseline
    }
    
    final_results = evaluate_models(models, test_loader, device)
    
    # Analysis and summary
    print("\n" + "="*80)
    print("🎉 SEQUENTIAL TRAINING RESULTS - THE ULTIMATE TEST")
    print("="*80)
    
    teacher_loss = final_results['Teacher (Joint Pipeline)']['test_loss']
    guided_loss = final_results['Student (Guided Sparse)']['test_loss']
    baseline_loss = final_results['Student (Baseline Sparse)']['test_loss']
    
    # Compare guided vs baseline student
    guidance_improvement = ((baseline_loss - guided_loss) / baseline_loss) * 100
    
    # Compare student vs teacher
    teacher_vs_guided = ((teacher_loss - guided_loss) / teacher_loss) * 100
    
    print(f"\n🏆 KEY FINDINGS:")
    print(f"   Teacher (Joint Pipeline): {teacher_loss:.4f} test loss")
    print(f"   Student (Guided Sparse): {guided_loss:.4f} test loss") 
    print(f"   Student (Baseline Sparse): {baseline_loss:.4f} test loss")
    
    print(f"\n📈 PERFORMANCE ANALYSIS:")
    print(f"   Guidance improvement: {guidance_improvement:+.1f}%")
    if guidance_improvement > 5:
        print(f"   ✅ BREAKTHROUGH: Teacher guidance significantly improves student!")
    elif guidance_improvement > 0:
        print(f"   🟡 Modest improvement from teacher guidance")
    else:
        print(f"   🔴 Teacher guidance didn't help")
    
    print(f"\n   Teacher vs guided student: {teacher_vs_guided:+.1f}%")
    if teacher_vs_guided < -5:
        print(f"   🚀 AMAZING: Student outperformed teacher!")
    elif teacher_vs_guided < 0:
        print(f"   ✅ SUCCESS: Student matched teacher performance") 
    else:
        print(f"   📚 Teacher still superior")
    
    # Efficiency analysis
    teacher_params = final_results['Teacher (Joint Pipeline)']['param_count']
    student_params = final_results['Student (Guided Sparse)']['param_count']
    efficiency_ratio = teacher_params / student_params
    
    print(f"\n⚡ EFFICIENCY ANALYSIS:")
    print(f"   Teacher parameters: {teacher_params:,}")
    print(f"   Student parameters: {student_params:,}")
    print(f"   Efficiency ratio: {efficiency_ratio:.2f}x")
    
    if guided_loss <= teacher_loss and efficiency_ratio > 1:
        print(f"   🏆 ULTIMATE SUCCESS: Student matches teacher with {efficiency_ratio:.1f}x efficiency!")
    
    # Save comprehensive results
    comprehensive_results = {
        'final_results': final_results,
        'training_results': {
            'teacher': teacher_results,
            'student_guided': student_guided_results,
            'student_baseline': student_baseline_results
        },
        'analysis': {
            'guidance_improvement': guidance_improvement,
            'teacher_vs_guided': teacher_vs_guided,
            'efficiency_ratio': efficiency_ratio
        }
    }
    
    with open(results_dir / 'sequential_training_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_dir / 'sequential_training_results.json'}")
    print(f"📚 Teacher guidance saved to {results_dir / 'teacher_guidance.pkl'}")
    
    return comprehensive_results


if __name__ == "__main__":
    results = main()
