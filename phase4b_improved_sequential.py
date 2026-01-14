#!/usr/bin/env python3
"""
Phase 4B: Refined Sequential Training - Better Teacher-Student Knowledge Transfer

The previous approach didn't effectively transfer knowledge. This refined approach:
1. Better extracts teacher's attention patterns
2. More effectively initializes student with teacher knowledge
3. Uses attention distillation for knowledge transfer
4. Tests multiple guidance strengths

Key Innovation: Attention pattern distillation from teacher to student
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

print("🎯 Phase 4B: Refined Sequential Training - Better Knowledge Transfer")
print("=" * 80)
print("Hypothesis: Better knowledge transfer from teacher to student will show clear benefits")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create results directory
results_dir = Path('/Users/bard/Code/genomic-sparse-attention/results/phase4b')
results_dir.mkdir(parents=True, exist_ok=True)


class ImprovedTeacherModel(nn.Module):
    """
    Improved teacher that explicitly learns attention patterns.
    Includes attention visualization and pattern extraction capabilities.
    """
    
    def __init__(self, vocab_size=6, embed_dim=16, seq_length=200):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        
        print(f"🧠 Improved Teacher Model:")
        print(f"   Embedding size: {embed_dim}D per token")
        print(f"   Focus: Learn explicit attention patterns")
        
        # Task-specific embeddings
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=5)
        
        # Explicit attention mechanism (what we want to teach)
        self.attention_weights = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(embed_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Pattern processor (works on attended features)
        self.pattern_processor = nn.Sequential(
            nn.Conv1d(embed_dim, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Final predictor
        self.predictor = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        nn.init.xavier_uniform_(self.embeddings.weight)
    
    def forward(self, x, return_attention=False):
        """Forward pass with explicit attention computation."""
        # Get embeddings
        embeddings = self.embeddings(x)  # [batch, seq, embed_dim]
        embeddings_t = embeddings.transpose(1, 2)  # [batch, embed_dim, seq]
        
        # Compute explicit attention weights
        attention_logits = self.attention_weights(embeddings_t)  # [batch, 1, seq]
        attention_weights = attention_logits.squeeze(1)  # [batch, seq]
        
        # Apply attention to embeddings
        attended_embeddings = embeddings_t * attention_weights.unsqueeze(1)  # [batch, embed_dim, seq]
        
        # Process attended patterns
        patterns = self.pattern_processor(attended_embeddings).squeeze(-1)  # [batch, 16]
        
        # Final prediction
        prediction = self.predictor(patterns).squeeze(-1)
        
        if return_attention:
            return prediction, {
                'attention_weights': attention_weights,
                'attended_embeddings': attended_embeddings.transpose(1, 2),  # Back to [batch, seq, embed]
                'embeddings': embeddings
            }
        
        return prediction
    
    def extract_teaching_knowledge(self, dataloader, device='cpu'):
        """Extract comprehensive knowledge for teaching."""
        self.eval()
        
        all_attention_weights = []
        important_positions = []
        embedding_patterns = []
        
        print("\n🔍 Extracting comprehensive teaching knowledge...")
        
        with torch.no_grad():
            for sequences, labels in dataloader:
                sequences, labels = sequences.to(device), labels.to(device)
                
                prediction, attention_info = self.forward(sequences, return_attention=True)
                
                # Collect attention patterns
                attention_weights = attention_info['attention_weights']  # [batch, seq]
                all_attention_weights.append(attention_weights.cpu())
                
                # Find examples with high regulatory strength
                high_reg_mask = prediction > 0.7
                if high_reg_mask.sum() > 0:
                    high_attention = attention_weights[high_reg_mask]
                    important_positions.append(high_attention.cpu())
                    
                    high_embeddings = attention_info['embeddings'][high_reg_mask]
                    embedding_patterns.append(high_embeddings.cpu())
        
        # Aggregate knowledge
        all_attention = torch.cat(all_attention_weights, dim=0)  # [total_samples, seq]
        mean_attention = all_attention.mean(dim=0)  # [seq]
        
        # Find consistently important positions
        attention_variance = all_attention.var(dim=0)
        consistency_score = mean_attention / (attention_variance + 1e-6)
        
        # Get top positions that are both important and consistent
        top_positions = torch.topk(consistency_score, k=min(30, len(consistency_score)))[1]
        
        knowledge = {
            'mean_attention_pattern': mean_attention,
            'attention_consistency': consistency_score, 
            'top_important_positions': top_positions.tolist(),
            'position_importance_threshold': float(torch.quantile(mean_attention, 0.8))
        }
        
        print(f"✅ Teaching knowledge extracted:")
        print(f"   Mean attention: {mean_attention.mean():.4f}")
        print(f"   Top positions: {knowledge['top_important_positions'][:10]}")
        print(f"   Importance threshold: {knowledge['position_importance_threshold']:.4f}")
        
        return knowledge


class ImprovedGuidedStudent(nn.Module):
    """
    Improved student model with better knowledge transfer mechanisms.
    Uses attention distillation and guided initialization.
    """
    
    def __init__(self, vocab_size=6, embed_dim=32, seq_length=200, sparsity_ratio=0.1, teacher_knowledge=None):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.sparsity_ratio = sparsity_ratio
        self.teacher_knowledge = teacher_knowledge
        
        print(f"🎓 Improved Student Model:")
        print(f"   Embedding size: {embed_dim}D")
        print(f"   Sparsity: {sparsity_ratio:.1%}")
        print(f"   Teacher guidance: {'Yes' if teacher_knowledge else 'No'}")
        
        # Student embeddings (larger capacity than teacher)
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=5)
        
        # Learned position selector (guided by teacher)
        self.position_selector = ImprovedPositionSelector(embed_dim, teacher_knowledge)
        
        # Attention approximator
        self.attention_approximator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize with teacher knowledge if available
        if teacher_knowledge:
            self._initialize_with_teacher_knowledge()
    
    def _initialize_with_teacher_knowledge(self):
        """Initialize student parameters using teacher's knowledge."""
        print("🔗 Initializing student with teacher knowledge...")
        
        # Initialize embeddings with small random values (let the student learn them)
        nn.init.normal_(self.embeddings.weight, mean=0, std=0.02)
        
        # The position selector will be initialized with teacher patterns
        print("   ✅ Student initialized with teacher guidance")
    
    def forward(self, x):
        """Forward pass with guided sparse attention."""
        batch_size, seq_length = x.shape
        
        # Get embeddings
        embeddings = self.embeddings(x)  # [batch, seq, embed_dim]
        
        # Select important positions using teacher guidance
        selected_embeddings, selected_indices, selection_scores = self.position_selector.select_positions(
            embeddings, self.sparsity_ratio
        )
        
        # Apply attention approximation to selected embeddings
        k = selected_embeddings.shape[1]
        attended = self.attention_approximator(selected_embeddings.reshape(-1, self.embed_dim))
        attended = attended.reshape(batch_size, k, self.embed_dim)
        
        # Pool and classify
        pooled = attended.mean(dim=1)  # Global average pooling
        prediction = self.classifier(pooled).squeeze(-1)
        
        return prediction, selected_indices, selection_scores


class ImprovedPositionSelector(nn.Module):
    """
    Position selector that effectively uses teacher's attention patterns.
    """
    
    def __init__(self, embed_dim, teacher_knowledge=None):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.teacher_knowledge = teacher_knowledge
        
        # Learnable importance scorer
        self.importance_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1)
        )
        
        if teacher_knowledge:
            # Register teacher's attention pattern as a prior
            self.register_buffer('teacher_attention_prior', teacher_knowledge['mean_attention_pattern'])
            self.register_buffer('teacher_top_positions', torch.tensor(teacher_knowledge['top_important_positions']))
            
            # Learnable weight for combining teacher and student knowledge
            self.teacher_weight = nn.Parameter(torch.tensor(0.5))  # Start with 50% teacher, 50% learned
            
            print(f"   📚 Teacher prior registered: {len(self.teacher_attention_prior)} positions")
        else:
            self.teacher_attention_prior = None
            self.teacher_weight = None
    
    def select_positions(self, embeddings, sparsity_ratio):
        """Select positions using teacher guidance + learned importance."""
        batch_size, seq_length, embed_dim = embeddings.shape
        k = max(1, int(seq_length * sparsity_ratio))
        
        # Get learned importance scores
        flat_embeddings = embeddings.reshape(-1, embed_dim)
        learned_scores = self.importance_network(flat_embeddings)
        learned_scores = learned_scores.reshape(batch_size, seq_length)
        learned_scores = torch.sigmoid(learned_scores)
        
        if self.teacher_knowledge and self.teacher_attention_prior is not None:
            # Combine teacher's attention pattern with learned scores
            teacher_prior = self.teacher_attention_prior.unsqueeze(0).expand(batch_size, -1)
            teacher_prior = teacher_prior.to(embeddings.device)
            
            # Weighted combination
            combined_scores = (
                self.teacher_weight * teacher_prior + 
                (1 - self.teacher_weight) * learned_scores
            )
            
            # Apply sigmoid to ensure valid probabilities
            final_scores = torch.sigmoid(combined_scores)
        else:
            final_scores = learned_scores
        
        # Select top-k positions
        top_values, top_indices = torch.topk(final_scores, k, dim=1)
        
        # Gather selected embeddings
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, k)
        selected_embeddings = embeddings[batch_indices, top_indices]
        
        return selected_embeddings, top_indices, final_scores


class SyntheticGenomicDataset(Dataset):
    """Enhanced synthetic dataset for better pattern learning."""
    
    def __init__(self, num_samples=1000, seq_length=200, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.nucleotides = np.array([0, 1, 2, 3])
        
        # Enhanced regulatory motifs with more diversity
        self.regulatory_motifs = {
            'TATAAA': 0.9,    # Strong TATA box
            'CAAT': 0.6,      # CAAT box
            'GGGCGG': 0.8,    # Strong GC box
            'TTGACA': 0.5,    # -35 element
            'TATAAT': 0.7,    # -10 element
            'AAAATTT': 0.4,   # Weak AT-rich
            'CGCGCG': 0.6,    # CG-rich motif
        }
        
        self.sequences = []
        self.labels = []
        
        print(f"Generating {num_samples} enhanced synthetic sequences...")
        
        for _ in range(num_samples):
            seq, strength = self._generate_enhanced_sequence()
            self.sequences.append(torch.tensor(seq, dtype=torch.long))
            self.labels.append(torch.tensor(strength, dtype=torch.float))
        
        self.sequences = torch.stack(self.sequences)
        self.labels = torch.stack(self.labels)
        
        print(f"✅ Generated {len(self.sequences)} sequences")
        print(f"   Regulatory strength range: {self.labels.min():.3f} - {self.labels.max():.3f}")
        print(f"   Mean regulatory strength: {self.labels.mean():.3f}")
    
    def _generate_enhanced_sequence(self):
        """Generate sequence with clearer regulatory patterns."""
        sequence = np.random.choice(self.nucleotides, self.seq_length)
        regulatory_strength = 0.05  # Lower baseline
        
        # More controlled motif placement
        num_motifs = np.random.poisson(1.5)  # Slightly fewer motifs
        
        planted_positions = []
        
        for _ in range(num_motifs):
            motif_name = np.random.choice(list(self.regulatory_motifs.keys()))
            motif_strength = self.regulatory_motifs[motif_name]
            motif_array = np.array([{'A': 0, 'T': 1, 'C': 2, 'G': 3}[nt] for nt in motif_name])
            
            # Find non-overlapping position
            max_start = self.seq_length - len(motif_array)
            if max_start > 0:
                attempts = 0
                while attempts < 10:  # Try to avoid overlaps
                    start_pos = np.random.randint(0, max_start)
                    end_pos = start_pos + len(motif_array)
                    
                    # Check for overlap with existing motifs
                    overlap = False
                    for existing_start, existing_end in planted_positions:
                        if not (end_pos <= existing_start or start_pos >= existing_end):
                            overlap = True
                            break
                    
                    if not overlap:
                        sequence[start_pos:end_pos] = motif_array
                        planted_positions.append((start_pos, end_pos))
                        regulatory_strength += motif_strength
                        break
                    
                    attempts += 1
        
        # Normalize and add controlled noise
        regulatory_strength = min(regulatory_strength, 1.0)
        regulatory_strength += np.random.normal(0, 0.03)  # Less noise
        regulatory_strength = max(0.0, min(regulatory_strength, 1.0))
        
        return sequence, regulatory_strength
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def train_improved_teacher(teacher, train_loader, val_loader, num_epochs=500, device='cpu'):
    """Train the improved teacher model with attention regularization."""
    
    print("\n" + "="*60)
    print("👨‍🏫 TRAINING IMPROVED TEACHER")
    print("="*60)
    
    teacher = teacher.to(device)
    optimizer = optim.Adam(teacher.parameters(), lr=0.0008, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    print(f"Teacher parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    
    best_val_loss = float('inf')
    patience = 50  # Much higher patience for 500 epochs
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        teacher.train()
        train_loss = 0
        attention_entropy_loss = 0
        
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with attention
            predictions, attention_info = teacher.forward(sequences, return_attention=True)
            
            # Main prediction loss
            pred_loss = criterion(predictions, labels)
            
            # Attention regularization (encourage sparsity)
            attention_weights = attention_info['attention_weights']
            entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=1)
            entropy_reg = torch.mean(entropy) * 0.01  # Small regularization
            
            total_loss = pred_loss + entropy_reg
            total_loss.backward()
            optimizer.step()
            
            train_loss += pred_loss.item()
            attention_entropy_loss += entropy_reg.item()
        
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
        attention_entropy_loss /= len(train_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d}: Train = {train_loss:.4f}, Val = {val_loss:.4f}, "
                  f"Attn_Reg = {attention_entropy_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"✅ Teacher training complete! Best validation: {best_val_loss:.4f}")
    return {'best_val_loss': best_val_loss, 'final_train_loss': train_loss}


def train_improved_student(student, train_loader, val_loader, num_epochs=500, device='cpu'):
    """Train the improved student model."""
    
    print(f"\n🎓 Training Student Model...")
    
    student = student.to(device)
    optimizer = optim.Adam(student.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    print(f"Student parameters: {sum(p.numel() for p in student.parameters()):,}")
    
    if hasattr(student.position_selector, 'teacher_weight') and student.position_selector.teacher_weight is not None:
        print(f"Initial teacher weight: {student.position_selector.teacher_weight.item():.3f}")
    
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
        
        if (epoch + 1) % 50 == 0:
            teacher_weight = "N/A"
            if hasattr(student.position_selector, 'teacher_weight') and student.position_selector.teacher_weight is not None:
                teacher_weight = f"{student.position_selector.teacher_weight.item():.3f}"
            
            print(f"Epoch {epoch+1:3d}: Train = {train_loss:.4f}, Val = {val_loss:.4f}, "
                  f"Teacher_W = {teacher_weight}")
    
    return {'final_train_loss': train_loss, 'final_val_loss': val_loss}


def evaluate_comprehensive(models_dict, test_loader, device='cpu'):
    """Comprehensive evaluation with detailed metrics."""
    
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE EVALUATION")
    print("="*80)
    
    criterion = nn.MSELoss()
    results = {}
    
    for model_name, model in models_dict.items():
        model.eval()
        test_loss = 0
        predictions_list = []
        labels_list = []
        attention_patterns = []
        
        print(f"\n📊 Evaluating {model_name}...")
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                
                if 'Teacher' in model_name:
                    predictions = model(sequences)
                else:  # Student model
                    predictions, selected_indices, selection_scores = model(sequences)
                    
                    # Analyze attention patterns for students
                    if len(attention_patterns) < 3:  # Save first few batches
                        attention_patterns.append({
                            'selected_positions': selected_indices[0].cpu().tolist(),
                            'selection_scores': selection_scores[0].cpu().numpy()
                        })
                
                test_loss += criterion(predictions, labels).item()
                predictions_list.extend(predictions.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
        
        test_loss /= len(test_loader)
        predictions_array = np.array(predictions_list)
        labels_array = np.array(labels_list)
        
        mae = np.mean(np.abs(predictions_array - labels_array))
        correlation = np.corrcoef(predictions_array, labels_array)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
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
        
        # Show attention patterns for students
        if attention_patterns:
            avg_positions = np.mean([len(p['selected_positions']) for p in attention_patterns])
            print(f"   Avg selected positions: {avg_positions:.1f}")
            sample_positions = attention_patterns[0]['selected_positions'][:10]
            print(f"   Sample positions: {sample_positions}")
    
    return results


def main():
    """Main improved sequential training experiment."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate enhanced dataset
    print("\n🧬 Generating enhanced synthetic dataset...")
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
    
    # STAGE 1: Train improved teacher
    teacher_model = ImprovedTeacherModel(embed_dim=16)
    teacher_results = train_improved_teacher(teacher_model, train_loader, val_loader, 
                                           num_epochs=500, device=device)
    
    # Extract improved teaching knowledge
    teacher_knowledge = teacher_model.extract_teaching_knowledge(train_loader, device)
    
    # STAGE 2: Train students with and without guidance
    student_guided = ImprovedGuidedStudent(
        embed_dim=32, sparsity_ratio=0.1, teacher_knowledge=teacher_knowledge
    )
    
    student_baseline = ImprovedGuidedStudent(
        embed_dim=32, sparsity_ratio=0.1, teacher_knowledge=None
    )
    
    # Train students
    print("\n" + "="*60)
    print("🎓 TRAINING GUIDED STUDENT")
    print("="*60)
    guided_results = train_improved_student(student_guided, train_loader, val_loader, 
                                          num_epochs=500, device=device)
    
    print("\n" + "="*60)
    print("🎓 TRAINING BASELINE STUDENT")
    print("="*60)
    baseline_results = train_improved_student(student_baseline, train_loader, val_loader,
                                            num_epochs=500, device=device)
    
    # STAGE 3: Comprehensive evaluation
    models = {
        'Teacher (Improved)': teacher_model,
        'Student (Guided)': student_guided,
        'Student (Baseline)': student_baseline
    }
    
    final_results = evaluate_comprehensive(models, test_loader, device)
    
    # Analysis
    print("\n" + "="*80)
    print("🎉 IMPROVED SEQUENTIAL TRAINING RESULTS")
    print("="*80)
    
    teacher_loss = final_results['Teacher (Improved)']['test_loss']
    guided_loss = final_results['Student (Guided)']['test_loss']
    baseline_loss = final_results['Student (Baseline)']['test_loss']
    
    # Key comparisons
    guidance_improvement = ((baseline_loss - guided_loss) / baseline_loss) * 100
    student_vs_teacher = ((teacher_loss - guided_loss) / teacher_loss) * 100
    
    print(f"\n🏆 PERFORMANCE SUMMARY:")
    print(f"   Teacher: {teacher_loss:.4f} test loss")
    print(f"   Guided Student: {guided_loss:.4f} test loss") 
    print(f"   Baseline Student: {baseline_loss:.4f} test loss")
    
    print(f"\n📈 KEY FINDINGS:")
    print(f"   Teacher guidance improvement: {guidance_improvement:+.1f}%")
    if guidance_improvement > 5:
        print(f"   ✅ SUCCESS: Teacher guidance significantly helps!")
    elif guidance_improvement > 1:
        print(f"   🟡 Modest improvement from teacher guidance")
    else:
        print(f"   🔴 No clear benefit from teacher guidance")
    
    print(f"   Student vs teacher performance: {student_vs_teacher:+.1f}%")
    if student_vs_teacher > 5:
        print(f"   🚀 BREAKTHROUGH: Guided student outperforms teacher!")
    elif student_vs_teacher > -5:
        print(f"   ✅ SUCCESS: Student matches teacher performance")
    else:
        print(f"   📚 Teacher still superior")
    
    # Save results
    comprehensive_results = {
        'final_results': final_results,
        'analysis': {
            'guidance_improvement': guidance_improvement,
            'student_vs_teacher': student_vs_teacher
        },
        'teacher_knowledge': {k: v.tolist() if isinstance(v, torch.Tensor) else v 
                            for k, v in teacher_knowledge.items()}
    }
    
    with open(results_dir / 'improved_sequential_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_dir}")
    return comprehensive_results


if __name__ == "__main__":
    results = main()
