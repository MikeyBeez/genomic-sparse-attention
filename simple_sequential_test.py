#!/usr/bin/env python3
"""
Simplified Sequential Training Experiment - Focus on Core Breakthrough
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime

# Ensure output is flushed
import sys
import os

print("🚀 Simplified Sequential Training Experiment - Core Breakthrough Test")
print("=" * 70)
print(f"Start time: {datetime.now()}")
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# Create results directory
results_dir = Path('/Users/bard/Code/genomic-sparse-attention/results/simple_sequential')
results_dir.mkdir(parents=True, exist_ok=True)

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

class SimpleTeacher(nn.Module):
    """Simple but effective teacher model."""
    
    def __init__(self, embed_dim=20):
        super().__init__()
        print(f"🧠 Simple Teacher: {embed_dim}D embeddings")
        
        self.embeddings = nn.Embedding(6, embed_dim, padding_idx=5)
        
        # Simple attention learning
        self.attention = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(embed_dim, 1, 5, padding=2),
            nn.Sigmoid()
        )
        
        # Pattern processor  
        self.processor = nn.Sequential(
            nn.Conv1d(embed_dim, 32, 7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, return_attention=False):
        emb = self.embeddings(x)
        emb_t = emb.transpose(1, 2)
        
        att_weights = self.attention(emb_t).squeeze(1)
        attended = emb_t * att_weights.unsqueeze(1)
        
        patterns = self.processor(attended).squeeze(-1)
        prediction = self.predictor(patterns).squeeze(-1)
        
        if return_attention:
            return prediction, {'attention_weights': att_weights}
        return prediction

class SimpleStudent(nn.Module):
    """Simple student with teacher guidance."""
    
    def __init__(self, embed_dim=20, teacher_knowledge=None):
        super().__init__()
        print(f"🎓 Simple Student: {embed_dim}D embeddings")
        
        self.embeddings = nn.Embedding(6, embed_dim, padding_idx=5)
        
        # Position selector
        self.selector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2),
            nn.ReLU(),
            nn.Linear(embed_dim//2, 1),
            nn.Sigmoid()
        )
        
        if teacher_knowledge:
            teacher_pattern = torch.tensor(teacher_knowledge['pattern'])
            self.register_buffer('teacher_guide', teacher_pattern)
            self.guide_weight = nn.Parameter(torch.tensor(0.6))
            print(f"   Teacher guidance loaded: {len(teacher_knowledge['top_pos'])} positions")
        
        # Processor
        self.processor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        emb = self.embeddings(x)
        batch_size, seq_len, embed_dim = emb.shape
        k = int(seq_len * 0.15)  # 15% sparsity
        
        # Get selection scores
        flat_emb = emb.reshape(-1, embed_dim)
        learned_scores = self.selector(flat_emb).reshape(batch_size, seq_len)
        
        if hasattr(self, 'teacher_guide'):
            teacher_scores = self.teacher_guide.unsqueeze(0).expand(batch_size, -1).to(emb.device)
            combined = self.guide_weight * teacher_scores + (1 - self.guide_weight) * learned_scores
            final_scores = torch.sigmoid(combined)
        else:
            final_scores = learned_scores
        
        # Select top-k
        _, indices = torch.topk(final_scores, k, dim=1)
        batch_idx = torch.arange(batch_size).unsqueeze(1).expand(-1, k)
        selected = emb[batch_idx, indices]
        
        # Process and predict
        pooled = selected.mean(dim=1)
        prediction = self.processor(pooled).squeeze(-1)
        
        return prediction, indices

class SimpleDataset(Dataset):
    """Simple but effective synthetic genomic dataset."""
    
    def __init__(self, num_samples=2000):
        print(f"🧬 Generating {num_samples} samples...")
        
        self.motifs = {
            'TATAAA': 0.8, 'CAAT': 0.6, 'GGGCGG': 0.7, 'TTGACA': 0.5
        }
        
        sequences = []
        labels = []
        
        for i in range(num_samples):
            seq = np.random.choice([0,1,2,3], 200)  # Random background
            strength = 0.1
            
            # Plant 1-3 motifs
            num_motifs = np.random.randint(1, 4)
            for _ in range(num_motifs):
                motif_name = np.random.choice(list(self.motifs.keys()))
                motif_strength = self.motifs[motif_name]
                motif_seq = [{'A':0,'T':1,'C':2,'G':3}[nt] for nt in motif_name]
                
                if len(motif_seq) < 190:
                    pos = np.random.randint(0, 200 - len(motif_seq))
                    seq[pos:pos+len(motif_seq)] = motif_seq
                    strength += motif_strength
            
            strength = min(strength + np.random.normal(0, 0.05), 1.0)
            strength = max(strength, 0.0)
            
            sequences.append(torch.tensor(seq, dtype=torch.long))
            labels.append(torch.tensor(strength, dtype=torch.float))
        
        self.sequences = torch.stack(sequences)
        self.labels = torch.stack(labels)
        
        print(f"✅ Dataset created: mean strength = {self.labels.mean():.3f}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def train_teacher(teacher, loader, epochs, device):
    """Train teacher model."""
    print(f"\n🏋️ Training Teacher ({epochs} epochs)...")
    
    teacher = teacher.to(device)
    optimizer = optim.Adam(teacher.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = teacher(sequences)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 50 == 0:
            avg_loss = total_loss / len(loader)
            print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    print("✅ Teacher training complete")

def extract_knowledge(teacher, loader, device):
    """Extract teacher knowledge."""
    print("\n🔍 Extracting teacher knowledge...")
    
    teacher.eval()
    all_attention = []
    all_pred = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            pred, info = teacher(sequences, return_attention=True)
            
            all_attention.append(info['attention_weights'].cpu())
            all_pred.append(pred.cpu())
            all_labels.append(labels.cpu())
    
    attention = torch.cat(all_attention)
    predictions = torch.cat(all_pred)  
    labels = torch.cat(all_labels)
    
    # Weight by prediction quality
    errors = torch.abs(predictions - labels)
    weights = torch.exp(-errors * 3)
    
    weighted_pattern = (attention * weights.unsqueeze(1)).sum(0) / weights.sum()
    top_positions = torch.topk(weighted_pattern, 30)[1].tolist()
    
    knowledge = {
        'pattern': weighted_pattern.tolist(),
        'top_pos': top_positions,
        'mean_error': float(errors.mean()),
        'correlation': float(np.corrcoef(predictions.numpy(), labels.numpy())[0,1])
    }
    
    print(f"✅ Knowledge extracted: correlation = {knowledge['correlation']:.3f}")
    return knowledge

def train_student(student, loader, epochs, device):
    """Train student model."""
    print(f"\n🎓 Training Student ({epochs} epochs)...")
    
    student = student.to(device)
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions, _ = student(sequences)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 25 == 0:
            avg_loss = total_loss / len(loader)
            print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    print("✅ Student training complete")

def evaluate_models(models, test_loader, device):
    """Evaluate all models."""
    print("\n📊 Evaluating models...")
    
    results = {}
    
    for name, model in models.items():
        model.eval()
        test_loss = 0
        predictions = []
        labels = []
        
        with torch.no_grad():
            for sequences, batch_labels in test_loader:
                sequences, batch_labels = sequences.to(device), batch_labels.to(device)
                
                if 'Teacher' in name:
                    preds = model(sequences)
                else:
                    preds, _ = model(sequences)
                
                test_loss += nn.MSELoss()(preds, batch_labels).item()
                predictions.extend(preds.cpu().numpy())
                labels.extend(batch_labels.cpu().numpy())
        
        test_loss /= len(test_loader)
        correlation = np.corrcoef(predictions, labels)[0,1]
        if np.isnan(correlation):
            correlation = 0.0
        
        results[name] = {
            'loss': test_loss,
            'correlation': correlation
        }
        
        print(f"  {name}: Loss={test_loss:.4f}, Correlation={correlation:.3f}")
    
    return results

def main():
    """Main simplified experiment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️ Using device: {device}")
    
    # Create dataset
    full_dataset = SimpleDataset(2000)
    
    # Split data
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))  
    test_size = len(full_dataset) - train_size - val_size
    
    train_data, val_data, test_data = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)
    
    print(f"📊 Data split: {train_size} train, {val_size} val, {test_size} test")
    
    # Create and train teacher
    teacher = SimpleTeacher()
    train_teacher(teacher, train_loader, 200, device)  # Reasonable epochs
    
    # Extract knowledge
    knowledge = extract_knowledge(teacher, train_loader, device)
    
    # Create students
    guided_student = SimpleStudent(teacher_knowledge=knowledge)
    baseline_student = SimpleStudent()
    
    # Train students
    train_student(guided_student, train_loader, 150, device)
    train_student(baseline_student, train_loader, 150, device)
    
    # Evaluate
    models = {
        'Teacher': teacher,
        'Guided Student': guided_student,
        'Baseline Student': baseline_student
    }
    
    results = evaluate_models(models, test_loader, device)
    
    # Analysis
    print("\n" + "="*60)
    print("🎉 SIMPLIFIED SEQUENTIAL TRAINING RESULTS")
    print("="*60)
    
    teacher_corr = results['Teacher']['correlation']
    guided_corr = results['Guided Student']['correlation']
    baseline_corr = results['Baseline Student']['correlation']
    
    guided_loss = results['Guided Student']['loss']
    baseline_loss = results['Baseline Student']['loss']
    
    improvement = ((baseline_loss - guided_loss) / baseline_loss) * 100 if baseline_loss > 0 else 0
    
    print(f"📊 RESULTS:")
    print(f"  Teacher correlation: {teacher_corr:.3f}")
    print(f"  Guided vs Baseline: {improvement:+.1f}% improvement")
    
    # Success evaluation
    teacher_success = teacher_corr > 0.2
    transfer_success = improvement > 2.0
    
    print(f"\n✅ SUCCESS CRITERIA:")
    print(f"  Teacher learned patterns: {'✅' if teacher_success else '❌'}")
    print(f"  Knowledge transfer worked: {'✅' if transfer_success else '❌'}")
    
    if teacher_success and transfer_success:
        print(f"\n🏆 BREAKTHROUGH VALIDATED!")
        print(f"Sequential training (joint → sparse) proven to work!")
    elif teacher_success:
        print(f"\n🟡 Partial success: Teacher learned, transfer needs work")
    else:
        print(f"\n🔄 Teacher needs more training")
    
    # Save results
    final_results = {
        'teacher_correlation': teacher_corr,
        'improvement_percent': improvement,
        'teacher_success': teacher_success,
        'transfer_success': transfer_success,
        'knowledge': knowledge,
        'completed_at': datetime.now().isoformat()
    }
    
    with open(results_dir / 'simple_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Results saved to {results_dir}")
    return final_results

if __name__ == "__main__":
    try:
        results = main()
        print("\n🎉 Experiment completed successfully!")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
