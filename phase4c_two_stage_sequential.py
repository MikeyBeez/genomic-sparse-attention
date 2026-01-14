#!/usr/bin/env python3
"""
Phase 4C: Proper Two-Stage Sequential Training

Correct Training Paradigm:
1. Stage 1: 500 epochs - Learn embeddings (representation learning)
2. Stage 2: Additional epochs - Learn task patterns using those embeddings  
3. Stage 3: Extract learned patterns from fully trained teacher
4. Stage 4: Transfer patterns to guide sparse attention training

This follows the proper deep learning training sequence.
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

print("🎯 Phase 4C: Proper Two-Stage Sequential Training")
print("=" * 80)
print("Stage 1: Learn embeddings (500 epochs)")
print("Stage 2: Learn task patterns (additional epochs)")
print("Stage 3: Extract and transfer knowledge")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create results directory
results_dir = Path('/Users/bard/Code/genomic-sparse-attention/results/phase4c')
results_dir.mkdir(parents=True, exist_ok=True)


class TwoStageTeacherModel(nn.Module):
    """
    Two-stage teacher model that separates embedding learning from task learning.
    """
    
    def __init__(self, vocab_size=6, embed_dim=16, seq_length=200):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        
        print(f"🧠 Two-Stage Teacher Model:")
        print(f"   Embedding size: {embed_dim}D per token")
        print(f"   Training: Stage 1 (embeddings) + Stage 2 (task)")
        
        # Embeddings - will be trained first
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=5)
        
        # Task-specific layers - will be trained second
        self.attention_detector = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim * 2, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(embed_dim * 2),
            nn.Conv1d(embed_dim * 2, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
        
        # Pattern processor for attended features
        self.pattern_processor = nn.Sequential(
            nn.Conv1d(embed_dim, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Final predictor
        self.predictor = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.embeddings.weight)
    
    def forward(self, x, return_attention=False):
        """Forward pass with optional attention extraction."""
        # Get embeddings
        embeddings = self.embeddings(x)  # [batch, seq, embed_dim]
        embeddings_t = embeddings.transpose(1, 2)  # [batch, embed_dim, seq]
        
        # Compute attention weights
        attention_weights = self.attention_detector(embeddings_t).squeeze(1)  # [batch, seq]
        
        # Apply attention
        attended = embeddings_t * attention_weights.unsqueeze(1)  # [batch, embed_dim, seq]
        
        # Process patterns
        patterns = self.pattern_processor(attended).squeeze(-1)  # [batch, 16]
        
        # Final prediction
        prediction = self.predictor(patterns).squeeze(-1)
        
        if return_attention:
            return prediction, {
                'attention_weights': attention_weights,
                'attended_embeddings': attended.transpose(1, 2),
                'raw_embeddings': embeddings
            }
        
        return prediction
    
    def extract_comprehensive_knowledge(self, dataloader, device='cpu'):
        """Extract comprehensive knowledge after full training."""
        self.eval()
        
        print("\n🔍 Extracting comprehensive knowledge from fully trained teacher...")
        
        all_attention_weights = []
        successful_patterns = []
        embedding_centroids = []
        
        with torch.no_grad():
            for sequences, labels in dataloader:
                sequences, labels = sequences.to(device), labels.to(device)
                
                prediction, attention_info = self.forward(sequences, return_attention=True)
                
                # Collect attention patterns
                attention_weights = attention_info['attention_weights']
                all_attention_weights.append(attention_weights.cpu())
                
                # Collect successful examples (high prediction, high actual)
                success_mask = (prediction > 0.6) & (labels > 0.6)
                if success_mask.sum() > 0:
                    successful_attention = attention_weights[success_mask]
                    successful_patterns.append(successful_attention.cpu())
                    
                    successful_embeddings = attention_info['raw_embeddings'][success_mask]
                    embedding_centroids.append(successful_embeddings.cpu())
        
        # Process collected data
        all_attention = torch.cat(all_attention_weights, dim=0)  # [total_samples, seq]
        
        # Find consistent attention patterns
        attention_mean = all_attention.mean(dim=0)  # [seq]
        attention_std = all_attention.std(dim=0)  # [seq]
        
        # Positions with high attention and low variance are most reliable
        consistency_score = attention_mean / (attention_std + 1e-6)
        
        # Get top reliable positions
        reliable_positions = torch.topk(consistency_score, k=min(40, len(consistency_score)))[1]
        high_attention_positions = torch.topk(attention_mean, k=min(30, len(attention_mean)))[1]
        
        # Combine and get unique positions
        important_positions = torch.unique(torch.cat([reliable_positions, high_attention_positions]))
        
        knowledge = {
            'attention_pattern': attention_mean,
            'attention_consistency': consistency_score,
            'important_positions': important_positions.tolist(),
            'attention_threshold': float(torch.quantile(attention_mean, 0.75)),
            'mean_attention': float(attention_mean.mean()),
            'max_attention': float(attention_mean.max())
        }
        
        print(f"✅ Knowledge extracted:")
        print(f"   Important positions ({len(important_positions)}): {important_positions[:15].tolist()}...")
        print(f"   Mean attention: {knowledge['mean_attention']:.4f}")
        print(f"   Attention threshold: {knowledge['attention_threshold']:.4f}")
        
        return knowledge


class KnowledgeGuidedStudent(nn.Module):
    """
    Student model that uses teacher's extracted knowledge effectively.
    """
    
    def __init__(self, vocab_size=6, embed_dim=32, seq_length=200, sparsity_ratio=0.1, teacher_knowledge=None):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.sparsity_ratio = sparsity_ratio
        self.teacher_knowledge = teacher_knowledge
        
        print(f"🎓 Knowledge-Guided Student:")
        print(f"   Embedding size: {embed_dim}D")
        print(f"   Sparsity: {sparsity_ratio:.1%}")
        print(f"   Teacher guidance: {'Yes' if teacher_knowledge else 'No'}")
        
        # Student embeddings (higher capacity)
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=5)
        
        # Knowledge-guided position selector
        self.position_selector = KnowledgeGuidedSelector(embed_dim, teacher_knowledge)
        
        # Attention approximator
        self.attention_approximator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Initialize embeddings
        nn.init.normal_(self.embeddings.weight, mean=0, std=0.02)
    
    def forward(self, x):
        """Forward pass with knowledge-guided sparse attention."""
        embeddings = self.embeddings(x)  # [batch, seq, embed_dim]
        
        # Use knowledge-guided selection
        selected_embeddings, selected_indices, selection_scores = self.position_selector.select_with_knowledge(
            embeddings, self.sparsity_ratio
        )
        
        # Apply attention approximation
        batch_size, k, embed_dim = selected_embeddings.shape
        attended = self.attention_approximator(selected_embeddings.reshape(-1, embed_dim))
        attended = attended.reshape(batch_size, k, embed_dim)
        
        # Pool and classify
        pooled = attended.mean(dim=1)
        prediction = self.classifier(pooled).squeeze(-1)
        
        return prediction, selected_indices, selection_scores


class KnowledgeGuidedSelector(nn.Module):
    """
    Position selector that effectively uses teacher's knowledge.
    """
    
    def __init__(self, embed_dim, teacher_knowledge=None):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.teacher_knowledge = teacher_knowledge
        
        # Learnable importance network
        self.importance_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, 1)
        )
        
        if teacher_knowledge:
            # Register teacher's patterns
            attention_pattern = torch.tensor(teacher_knowledge['attention_pattern'], dtype=torch.float32)
            self.register_buffer('teacher_attention', attention_pattern)
            
            # Important positions as a binary mask
            important_pos = teacher_knowledge['important_positions']
            position_mask = torch.zeros(200)  # seq_length
            position_mask[important_pos] = 1.0
            self.register_buffer('important_positions_mask', position_mask)
            
            # Learnable combination weights
            self.knowledge_weight = nn.Parameter(torch.tensor(0.4))  # Start with 40% teacher knowledge
            self.position_boost = nn.Parameter(torch.tensor(2.0))   # Boost factor for important positions
            
            print(f"   📚 Teacher knowledge loaded: {len(important_pos)} important positions")
        else:
            self.teacher_attention = None
            self.knowledge_weight = None
    
    def select_with_knowledge(self, embeddings, sparsity_ratio):
        """Select positions using teacher knowledge and learned patterns."""
        batch_size, seq_length, embed_dim = embeddings.shape
        k = max(1, int(seq_length * sparsity_ratio))
        
        # Get learned importance scores
        flat_embeddings = embeddings.reshape(-1, embed_dim)
        learned_scores = self.importance_net(flat_embeddings).reshape(batch_size, seq_length)
        learned_scores = torch.sigmoid(learned_scores)
        
        if self.teacher_knowledge and self.teacher_attention is not None:
            # Get teacher's attention pattern
            teacher_scores = self.teacher_attention.unsqueeze(0).expand(batch_size, -1).to(embeddings.device)
            
            # Boost scores for known important positions
            position_boost = self.important_positions_mask.unsqueeze(0).expand(batch_size, -1).to(embeddings.device)
            boosted_learned = learned_scores + (self.position_boost * position_boost * learned_scores)
            
            # Combine teacher and boosted learned scores
            combined_scores = (
                self.knowledge_weight * teacher_scores + 
                (1 - self.knowledge_weight) * boosted_learned
            )
            
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
    """Synthetic genomic dataset for two-stage training."""
    
    def __init__(self, num_samples=1000, seq_length=200, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.nucleotides = np.array([0, 1, 2, 3])
        
        # Regulatory motifs
        self.regulatory_motifs = {
            'TATAAA': 0.9,    # Strong TATA box
            'CAAT': 0.6,      # CAAT box
            'GGGCGG': 0.8,    # GC box
            'TTGACA': 0.5,    # -35 element
            'TATAAT': 0.7,    # -10 element
            'AAAATTT': 0.4,   # AT-rich
            'CGCGCG': 0.6,    # CG-rich
        }
        
        self.sequences = []
        self.labels = []
        
        print(f"Generating {num_samples} sequences for two-stage training...")
        
        for _ in range(num_samples):
            seq, strength = self._generate_sequence()
            self.sequences.append(torch.tensor(seq, dtype=torch.long))
            self.labels.append(torch.tensor(strength, dtype=torch.float))
        
        self.sequences = torch.stack(self.sequences)
        self.labels = torch.stack(self.labels)
        
        print(f"✅ Generated {len(self.sequences)} sequences")
        print(f"   Regulatory strength: {self.labels.min():.3f} - {self.labels.max():.3f} (mean: {self.labels.mean():.3f})")
    
    def _generate_sequence(self):
        """Generate sequence with planted regulatory motifs."""
        sequence = np.random.choice(self.nucleotides, self.seq_length)
        regulatory_strength = 0.05  # Base level
        
        # Plant motifs with some randomness
        num_motifs = np.random.poisson(1.8)
        planted_positions = []
        
        for _ in range(num_motifs):
            motif_name = np.random.choice(list(self.regulatory_motifs.keys()))
            motif_strength = self.regulatory_motifs[motif_name]
            motif_array = np.array([{'A': 0, 'T': 1, 'C': 2, 'G': 3}[nt] for nt in motif_name])
            
            max_start = self.seq_length - len(motif_array)
            if max_start > 0:
                # Try to avoid overlaps
                for attempt in range(15):
                    start_pos = np.random.randint(0, max_start)
                    end_pos = start_pos + len(motif_array)
                    
                    # Check for overlap
                    overlap = any(not (end_pos <= existing[0] or start_pos >= existing[1]) 
                                 for existing in planted_positions)
                    
                    if not overlap:
                        sequence[start_pos:end_pos] = motif_array
                        planted_positions.append((start_pos, end_pos))
                        regulatory_strength += motif_strength
                        break
        
        # Add controlled noise and normalize
        regulatory_strength = min(regulatory_strength, 1.0)
        regulatory_strength += np.random.normal(0, 0.04)
        regulatory_strength = max(0.0, min(regulatory_strength, 1.0))
        
        return sequence, regulatory_strength
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def train_stage1_embeddings(teacher, train_loader, num_epochs=500, device='cpu'):
    """Stage 1: Train embeddings only (representation learning)."""
    
    print("\n" + "="*60)
    print("📚 STAGE 1: EMBEDDING LEARNING (500 epochs)")
    print("="*60)
    print("Goal: Learn good token representations")
    
    teacher = teacher.to(device)
    
    # Only train embeddings in stage 1
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.embeddings.weight.requires_grad = True
    
    optimizer = optim.Adam([teacher.embeddings.weight], lr=0.001)
    criterion = nn.MSELoss()
    
    print(f"Stage 1 parameters: {sum(p.numel() for p in teacher.parameters() if p.requires_grad):,}")
    
    for epoch in range(num_epochs):
        teacher.train()
        total_loss = 0
        
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = teacher(sequences)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1:3d}: Embedding Loss = {avg_loss:.4f}")
    
    print("✅ Stage 1 complete: Embeddings learned")
    return teacher


def train_stage2_task(teacher, train_loader, val_loader, num_epochs=200, device='cpu'):
    """Stage 2: Train task-specific layers using learned embeddings."""
    
    print("\n" + "="*60)
    print("🎯 STAGE 2: TASK LEARNING (200 epochs)")
    print("="*60)
    print("Goal: Learn regulatory prediction using learned embeddings")
    
    # Unfreeze all parameters for task learning
    for param in teacher.parameters():
        param.requires_grad = True
    
    # Use smaller learning rate for embeddings, normal for task layers
    embedding_params = [teacher.embeddings.weight]
    task_params = [p for n, p in teacher.named_parameters() if 'embeddings' not in n]
    
    optimizer = optim.Adam([
        {'params': embedding_params, 'lr': 0.0001},  # Smaller LR for embeddings
        {'params': task_params, 'lr': 0.001}         # Normal LR for task layers
    ])
    
    criterion = nn.MSELoss()
    
    print(f"Stage 2 parameters: {sum(p.numel() for p in teacher.parameters()):,}")
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
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
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1:3d}: Train = {train_loss:.4f}, Val = {val_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"✅ Stage 2 complete: Best validation loss = {best_val_loss:.4f}")
    return {'best_val_loss': best_val_loss, 'final_train_loss': train_loss}


def train_knowledge_guided_student(student, train_loader, val_loader, num_epochs=300, device='cpu'):
    """Train student with teacher's knowledge."""
    
    print(f"\n🎓 Training Knowledge-Guided Student ({num_epochs} epochs)...")
    
    student = student.to(device)
    optimizer = optim.Adam(student.parameters(), lr=0.0008, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    print(f"Student parameters: {sum(p.numel() for p in student.parameters()):,}")
    
    if hasattr(student.position_selector, 'knowledge_weight'):
        print(f"Initial knowledge weight: {student.position_selector.knowledge_weight.item():.3f}")
    
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
            knowledge_weight = "N/A"
            if hasattr(student.position_selector, 'knowledge_weight'):
                knowledge_weight = f"{student.position_selector.knowledge_weight.item():.3f}"
            
            print(f"Epoch {epoch+1:3d}: Train = {train_loss:.4f}, Val = {val_loss:.4f}, Knowledge_W = {knowledge_weight}")
    
    return {'final_train_loss': train_loss, 'final_val_loss': val_loss}


def evaluate_final_models(models_dict, test_loader, device='cpu'):
    """Final comprehensive evaluation."""
    
    print("\n" + "="*80)
    print("🎯 FINAL EVALUATION - Two-Stage Training Results")
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
                else:  # Student
                    predictions, selected_indices, _ = model(sequences)
                    
                    # Show selection pattern for first batch
                    if test_loss == 0:  # First batch
                        sample_positions = selected_indices[0].cpu().tolist()[:10]
                        print(f"   Sample selected positions: {sample_positions}")
                
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
    
    return results


def main():
    """Main two-stage sequential training experiment."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate dataset
    print("\n🧬 Generating dataset for two-stage training...")
    full_dataset = SyntheticGenomicDataset(num_samples=2200, seq_length=200, seed=42)
    
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
    
    # Initialize teacher model
    teacher_model = TwoStageTeacherModel(embed_dim=20)
    
    # STAGE 1: Learn embeddings (500 epochs)
    teacher_model = train_stage1_embeddings(teacher_model, train_loader, 
                                           num_epochs=500, device=device)
    
    # STAGE 2: Learn task using embeddings (200 epochs)
    task_results = train_stage2_task(teacher_model, train_loader, val_loader,
                                   num_epochs=200, device=device)
    
    # STAGE 3: Extract knowledge from fully trained teacher
    teacher_knowledge = teacher_model.extract_comprehensive_knowledge(train_loader, device)
    
    # STAGE 4: Train students with and without knowledge
    student_guided = KnowledgeGuidedStudent(
        embed_dim=32, sparsity_ratio=0.12, teacher_knowledge=teacher_knowledge
    )
    
    student_baseline = KnowledgeGuidedStudent(
        embed_dim=32, sparsity_ratio=0.12, teacher_knowledge=None
    )
    
    print("\n" + "="*60)
    print("🎓 TRAINING KNOWLEDGE-GUIDED STUDENT")
    print("="*60)
    guided_results = train_knowledge_guided_student(student_guided, train_loader, val_loader,
                                                   num_epochs=300, device=device)
    
    print("\n" + "="*60)
    print("🎓 TRAINING BASELINE STUDENT")
    print("="*60)
    baseline_results = train_knowledge_guided_student(student_baseline, train_loader, val_loader,
                                                     num_epochs=300, device=device)
    
    # Final evaluation
    models = {
        'Teacher (Two-Stage)': teacher_model,
        'Student (Knowledge-Guided)': student_guided,
        'Student (Baseline)': student_baseline
    }
    
    final_results = evaluate_final_models(models, test_loader, device)
    
    # Analysis
    print("\n" + "="*80)
    print("🎉 TWO-STAGE SEQUENTIAL TRAINING RESULTS")
    print("="*80)
    
    teacher_loss = final_results['Teacher (Two-Stage)']['test_loss']
    guided_loss = final_results['Student (Knowledge-Guided)']['test_loss']
    baseline_loss = final_results['Student (Baseline)']['test_loss']
    
    teacher_corr = final_results['Teacher (Two-Stage)']['correlation']
    guided_corr = final_results['Student (Knowledge-Guided)']['correlation']
    baseline_corr = final_results['Student (Baseline)']['correlation']
    
    # Key metrics
    knowledge_transfer_improvement = ((baseline_loss - guided_loss) / baseline_loss) * 100
    student_vs_teacher_performance = ((teacher_loss - guided_loss) / teacher_loss) * 100
    
    print(f"\n🏆 PERFORMANCE SUMMARY:")
    print(f"   Teacher: {teacher_loss:.4f} loss, {teacher_corr:.3f} correlation")
    print(f"   Knowledge-Guided Student: {guided_loss:.4f} loss, {guided_corr:.3f} correlation")
    print(f"   Baseline Student: {baseline_loss:.4f} loss, {baseline_corr:.3f} correlation")
    
    print(f"\n📈 KEY FINDINGS:")
    print(f"   Knowledge transfer improvement: {knowledge_transfer_improvement:+.1f}%")
    
    if knowledge_transfer_improvement > 5:
        print(f"   ✅ SUCCESS: Teacher knowledge significantly helps student!")
    elif knowledge_transfer_improvement > 1:
        print(f"   🟡 Modest benefit from teacher knowledge")
    else:
        print(f"   🔴 No clear knowledge transfer benefit")
    
    print(f"   Student vs teacher: {student_vs_teacher_performance:+.1f}%")
    
    if guided_corr > 0.3:
        print(f"   ✅ Student learned meaningful patterns (correlation > 0.3)")
    elif guided_corr > 0.1:
        print(f"   🟡 Student learned some patterns (correlation > 0.1)")
    else:
        print(f"   🔴 Student failed to learn meaningful patterns")
    
    # Save comprehensive results
    comprehensive_results = {
        'final_results': final_results,
        'training_results': {
            'task_training': task_results,
            'guided_student': guided_results,
            'baseline_student': baseline_results
        },
        'teacher_knowledge': teacher_knowledge,
        'analysis': {
            'knowledge_transfer_improvement': knowledge_transfer_improvement,
            'student_vs_teacher': student_vs_teacher_performance,
            'teacher_learned_patterns': teacher_corr > 0.3,
            'student_learned_patterns': guided_corr > 0.3
        }
    }
    
    with open(results_dir / 'two_stage_sequential_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\n💾 Complete results saved to {results_dir}")
    
    return comprehensive_results


if __name__ == "__main__":
    results = main()
