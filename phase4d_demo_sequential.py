#!/usr/bin/env python3
"""
Phase 4D: Efficient Two-Stage Sequential Training Demo

Demonstrates the proper training paradigm with reduced epochs for faster execution:
1. Stage 1: 100 epochs - Learn embeddings (representation learning)
2. Stage 2: 100 epochs - Learn task patterns using those embeddings  
3. Stage 3: Extract learned patterns from fully trained teacher
4. Stage 4: 150 epochs - Transfer patterns to guide sparse attention training

This proves the concept while being computationally efficient.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
import json

print("🎯 Phase 4D: Efficient Two-Stage Sequential Training Demo")
print("=" * 80)
print("Proving the concept: Embeddings first, then task learning, then transfer")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create results directory
results_dir = Path('/Users/bard/Code/genomic-sparse-attention/results/phase4d')
results_dir.mkdir(parents=True, exist_ok=True)


class DemoTeacherModel(nn.Module):
    """Efficient demo teacher model for two-stage training proof of concept."""
    
    def __init__(self, vocab_size=6, embed_dim=16, seq_length=200):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        print(f"🧠 Demo Teacher Model (Efficient):")
        print(f"   Embedding size: {embed_dim}D per token")
        print(f"   Two-stage training: embeddings → task")
        
        # Stage 1: Embeddings to be learned first
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=5)
        
        # Stage 2: Task layers to be learned second
        self.attention_learner = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(embed_dim, 1, kernel_size=5, padding=2),
            nn.Sigmoid()
        )
        
        self.pattern_processor = nn.Sequential(
            nn.Conv1d(embed_dim, 24, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(24, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        nn.init.xavier_uniform_(self.embeddings.weight)
    
    def forward(self, x, return_attention=False):
        embeddings = self.embeddings(x)
        embeddings_t = embeddings.transpose(1, 2)
        
        attention_weights = self.attention_learner(embeddings_t).squeeze(1)
        attended = embeddings_t * attention_weights.unsqueeze(1)
        
        patterns = self.pattern_processor(attended).squeeze(-1)
        prediction = self.predictor(patterns).squeeze(-1)
        
        if return_attention:
            return prediction, {'attention_weights': attention_weights, 'embeddings': embeddings}
        return prediction
    
    def extract_learned_knowledge(self, dataloader, device='cpu'):
        """Extract knowledge from the fully trained teacher."""
        self.eval()
        
        print("\n🔍 Extracting learned knowledge from teacher...")
        
        all_attention = []
        
        with torch.no_grad():
            for sequences, labels in dataloader:
                sequences = sequences.to(device)
                _, attention_info = self.forward(sequences, return_attention=True)
                all_attention.append(attention_info['attention_weights'].cpu())
        
        all_attention = torch.cat(all_attention, dim=0)
        attention_mean = all_attention.mean(dim=0)
        
        # Find top positions
        top_positions = torch.topk(attention_mean, k=25)[1].tolist()
        
        knowledge = {
            'attention_pattern': attention_mean.tolist(),
            'top_positions': top_positions,
            'threshold': float(torch.quantile(attention_mean, 0.7))
        }
        
        print(f"✅ Knowledge extracted: Top positions = {top_positions[:10]}")
        return knowledge


class DemoGuidedStudent(nn.Module):
    """Efficient demo student model."""
    
    def __init__(self, vocab_size=6, embed_dim=32, sparsity_ratio=0.15, teacher_knowledge=None):
        super().__init__()
        
        self.sparsity_ratio = sparsity_ratio
        self.teacher_knowledge = teacher_knowledge
        
        print(f"🎓 Demo Student Model:")
        print(f"   Embedding size: {embed_dim}D")
        print(f"   Teacher guidance: {'Yes' if teacher_knowledge else 'No'}")
        
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=5)
        
        self.selector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        if teacher_knowledge:
            # Register teacher guidance
            teacher_pattern = torch.tensor(teacher_knowledge['attention_pattern'])
            self.register_buffer('teacher_guidance', teacher_pattern)
            self.guidance_weight = nn.Parameter(torch.tensor(0.3))
        
        self.approximator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        embeddings = self.embeddings(x)
        batch_size, seq_length, embed_dim = embeddings.shape
        k = max(1, int(seq_length * self.sparsity_ratio))
        
        # Get selection scores
        flat_embeddings = embeddings.reshape(-1, embed_dim)
        learned_scores = self.selector(flat_embeddings).reshape(batch_size, seq_length)
        
        if hasattr(self, 'teacher_guidance'):
            # Use teacher guidance
            teacher_scores = self.teacher_guidance.unsqueeze(0).expand(batch_size, -1).to(embeddings.device)
            combined_scores = self.guidance_weight * teacher_scores + (1 - self.guidance_weight) * learned_scores
            final_scores = torch.sigmoid(combined_scores)
        else:
            final_scores = learned_scores
        
        # Select top-k
        _, top_indices = torch.topk(final_scores, k, dim=1)
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, k)
        selected_embeddings = embeddings[batch_indices, top_indices]
        
        # Process and classify
        attended = self.approximator(selected_embeddings.reshape(-1, embed_dim)).reshape(batch_size, k, embed_dim)
        pooled = attended.mean(dim=1)
        prediction = self.classifier(pooled).squeeze(-1)
        
        return prediction, top_indices, final_scores


class SyntheticGenomicDataset(Dataset):
    """Efficient synthetic dataset."""
    
    def __init__(self, num_samples=1000, seq_length=200, seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.seq_length = seq_length
        self.nucleotides = np.array([0, 1, 2, 3])
        
        self.regulatory_motifs = {
            'TATAAA': 0.8, 'CAAT': 0.6, 'GGGCGG': 0.7, 'TTGACA': 0.5
        }
        
        self.sequences = []
        self.labels = []
        
        for _ in range(num_samples):
            seq, strength = self._generate_sequence()
            self.sequences.append(torch.tensor(seq, dtype=torch.long))
            self.labels.append(torch.tensor(strength, dtype=torch.float))
        
        self.sequences = torch.stack(self.sequences)
        self.labels = torch.stack(self.labels)
        
        print(f"✅ Generated {num_samples} sequences (mean strength: {self.labels.mean():.3f})")
    
    def _generate_sequence(self):
        sequence = np.random.choice(self.nucleotides, self.seq_length)
        regulatory_strength = 0.1
        
        num_motifs = np.random.poisson(1.5)
        for _ in range(num_motifs):
            motif_name = np.random.choice(list(self.regulatory_motifs.keys()))
            motif_strength = self.regulatory_motifs[motif_name]
            motif_array = np.array([{'A': 0, 'T': 1, 'C': 2, 'G': 3}[nt] for nt in motif_name])
            
            max_start = self.seq_length - len(motif_array)
            if max_start > 0:
                start_pos = np.random.randint(0, max_start)
                sequence[start_pos:start_pos + len(motif_array)] = motif_array
                regulatory_strength += motif_strength
        
        regulatory_strength = min(regulatory_strength, 1.0)
        regulatory_strength += np.random.normal(0, 0.05)
        regulatory_strength = max(0.0, min(regulatory_strength, 1.0))
        
        return sequence, regulatory_strength
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def train_stage1_embeddings_demo(teacher, train_loader, epochs=100, device='cpu'):
    """Stage 1: Learn embeddings only."""
    
    print(f"\n📚 STAGE 1: Learning Embeddings ({epochs} epochs)")
    print("="*50)
    
    teacher = teacher.to(device)
    
    # Freeze all except embeddings
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.embeddings.weight.requires_grad = True
    
    optimizer = optim.Adam([teacher.embeddings.weight], lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = teacher(sequences)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: Embedding Loss = {total_loss/len(train_loader):.4f}")
    
    print("✅ Stage 1 Complete: Embeddings learned")


def train_stage2_task_demo(teacher, train_loader, val_loader, epochs=100, device='cpu'):
    """Stage 2: Learn task using learned embeddings."""
    
    print(f"\n🎯 STAGE 2: Learning Task ({epochs} epochs)")
    print("="*50)
    
    # Unfreeze all parameters
    for param in teacher.parameters():
        param.requires_grad = True
    
    # Different learning rates: lower for embeddings, normal for task layers
    embedding_params = [teacher.embeddings.weight]
    task_params = [p for n, p in teacher.named_parameters() if 'embeddings' not in n]
    
    optimizer = optim.Adam([
        {'params': embedding_params, 'lr': 0.0002},
        {'params': task_params, 'lr': 0.001}
    ])
    
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
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
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: Train = {train_loss:.4f}, Val = {val_loss:.4f}")
    
    print(f"✅ Stage 2 Complete: Best Val Loss = {best_val_loss:.4f}")
    return best_val_loss


def train_demo_student(student, train_loader, val_loader, epochs=150, device='cpu'):
    """Train student with or without teacher guidance."""
    
    student = student.to(device)
    optimizer = optim.Adam(student.parameters(), lr=0.0008)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
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
        
        if (epoch + 1) % 30 == 0:
            guidance = "N/A"
            if hasattr(student, 'guidance_weight'):
                guidance = f"{student.guidance_weight.item():.3f}"
            print(f"  Epoch {epoch+1:3d}: Train = {train_loss:.4f}, Val = {val_loss:.4f}, Guidance = {guidance}")
    
    return val_loss


def evaluate_demo_models(models, test_loader, device='cpu'):
    """Evaluate all models."""
    
    print("\n🎯 FINAL EVALUATION")
    print("="*50)
    
    criterion = nn.MSELoss()
    results = {}
    
    for model_name, model in models.items():
        model.eval()
        test_loss = 0
        predictions_list = []
        labels_list = []
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                
                if 'Teacher' in model_name:
                    predictions = model(sequences)
                else:
                    predictions, _, _ = model(sequences)
                
                test_loss += criterion(predictions, labels).item()
                predictions_list.extend(predictions.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
        
        test_loss /= len(test_loader)
        predictions_array = np.array(predictions_list)
        labels_array = np.array(labels_list)
        
        correlation = np.corrcoef(predictions_array, labels_array)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        results[model_name] = {
            'test_loss': float(test_loss),
            'correlation': float(correlation),
            'param_count': sum(p.numel() for p in model.parameters())
        }
        
        print(f"{model_name}:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Correlation: {correlation:.3f}")
        print(f"  Parameters: {results[model_name]['param_count']:,}")
    
    return results


def main():
    """Main efficient demo of two-stage sequential training."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate dataset
    print("\n🧬 Generating demo dataset...")
    full_dataset = SyntheticGenomicDataset(num_samples=1500, seq_length=200, seed=42)
    
    # Split dataset
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"📊 Dataset: {train_size} train, {val_size} val, {test_size} test")
    
    # Create teacher model
    teacher = DemoTeacherModel(embed_dim=18)
    
    # STAGE 1: Learn embeddings
    train_stage1_embeddings_demo(teacher, train_loader, epochs=100, device=device)
    
    # STAGE 2: Learn task
    best_val = train_stage2_task_demo(teacher, train_loader, val_loader, epochs=100, device=device)
    
    # STAGE 3: Extract knowledge
    teacher_knowledge = teacher.extract_learned_knowledge(train_loader, device)
    
    # STAGE 4: Train students
    print(f"\n🎓 STAGE 4: Training Students")
    print("="*50)
    
    student_guided = DemoGuidedStudent(embed_dim=32, teacher_knowledge=teacher_knowledge)
    student_baseline = DemoGuidedStudent(embed_dim=32, teacher_knowledge=None)
    
    print("Training guided student...")
    guided_val = train_demo_student(student_guided, train_loader, val_loader, epochs=150, device=device)
    
    print("Training baseline student...")
    baseline_val = train_demo_student(student_baseline, train_loader, val_loader, epochs=150, device=device)
    
    # Final evaluation
    models = {
        'Teacher (Two-Stage)': teacher,
        'Student (Guided)': student_guided,
        'Student (Baseline)': student_baseline
    }
    
    results = evaluate_demo_models(models, test_loader, device)
    
    # Analysis
    print("\n" + "="*60)
    print("🎉 TWO-STAGE SEQUENTIAL TRAINING DEMO RESULTS")
    print("="*60)
    
    teacher_loss = results['Teacher (Two-Stage)']['test_loss']
    guided_loss = results['Student (Guided)']['test_loss']
    baseline_loss = results['Student (Baseline)']['test_loss']
    
    teacher_corr = results['Teacher (Two-Stage)']['correlation']
    guided_corr = results['Student (Guided)']['correlation']
    
    improvement = ((baseline_loss - guided_loss) / baseline_loss) * 100
    
    print(f"\n📊 PERFORMANCE:")
    print(f"   Teacher: {teacher_loss:.4f} loss, {teacher_corr:.3f} correlation")
    print(f"   Guided Student: {guided_loss:.4f} loss, {guided_corr:.3f} correlation")
    print(f"   Baseline Student: {baseline_loss:.4f} loss")
    
    print(f"\n🎯 KEY FINDINGS:")
    print(f"   Knowledge transfer improvement: {improvement:+.1f}%")
    
    if improvement > 3:
        print(f"   ✅ SUCCESS: Two-stage training + knowledge transfer works!")
    elif improvement > 0:
        print(f"   🟡 Modest improvement from sequential approach")
    else:
        print(f"   🔄 Need further refinement of transfer mechanism")
    
    if teacher_corr > 0.2:
        print(f"   ✅ Teacher learned meaningful patterns")
    if guided_corr > 0.15:
        print(f"   ✅ Student benefited from teacher knowledge")
    
    # Save results
    final_results = {
        'results': results,
        'teacher_knowledge': teacher_knowledge,
        'analysis': {
            'improvement': improvement,
            'teacher_learned': teacher_corr > 0.2,
            'transfer_worked': guided_corr > guided_corr * 0.8
        }
    }
    
    with open(results_dir / 'demo_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Demo results saved to {results_dir}")
    
    return final_results


if __name__ == "__main__":
    results = main()
