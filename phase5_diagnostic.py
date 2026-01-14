#!/usr/bin/env python3
"""
Diagnostic version of the definitive sequential training experiment.
This version includes more immediate output and error handling.
"""
import sys
import os
import traceback
from datetime import datetime

print("🚀 Phase 5 Diagnostic: Starting Sequential Training Experiment", flush=True)
print(f"⏰ Start time: {datetime.now()}", flush=True)
print(f"🐍 Python version: {sys.version}", flush=True)
print(f"📁 Working directory: {os.getcwd()}", flush=True)

try:
    print("📦 Importing torch...", flush=True)
    import torch
    print(f"✅ PyTorch version: {torch.__version__}", flush=True)
    print(f"🖥️ Device available: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}", flush=True)
    
    print("📦 Importing other modules...", flush=True)
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    from pathlib import Path
    import json
    import time
    
    print("✅ All imports successful", flush=True)
    
    # Test basic functionality
    print("🧪 Testing basic PyTorch functionality...", flush=True)
    test_tensor = torch.randn(10, 10)
    test_result = torch.matmul(test_tensor, test_tensor.T)
    print(f"✅ Basic tensor operations work: {test_result.shape}", flush=True)
    
    # Set seeds
    print("🎲 Setting random seeds...", flush=True)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test dataset creation
    print("🧬 Testing dataset creation...", flush=True)
    
    class TestDataset(Dataset):
        def __init__(self, num_samples=100):
            print(f"   Creating test dataset with {num_samples} samples...", flush=True)
            self.sequences = torch.randint(0, 4, (num_samples, 200))
            self.labels = torch.rand(num_samples)
            print(f"   ✅ Test dataset created successfully", flush=True)
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            return self.sequences[idx], self.labels[idx]
    
    test_dataset = TestDataset(100)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    print("🧠 Testing simple model creation...", flush=True)
    
    class SimpleTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(6, 16, padding_idx=5)
            self.conv = nn.Conv1d(16, 32, kernel_size=5, padding=2)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Linear(32, 1)
        
        def forward(self, x):
            emb = self.embedding(x)  # [batch, seq, embed]
            emb_t = emb.transpose(1, 2)  # [batch, embed, seq]
            conv_out = torch.relu(self.conv(emb_t))
            pooled = self.pool(conv_out).squeeze(-1)  # [batch, 32]
            return torch.sigmoid(self.classifier(pooled)).squeeze(-1)
    
    test_model = SimpleTestModel()
    print(f"✅ Simple model created with {sum(p.numel() for p in test_model.parameters())} parameters", flush=True)
    
    # Test training loop
    print("🏋️ Testing training loop...", flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_model = test_model.to(device)
    optimizer = torch.optim.Adam(test_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("   Starting mini training test...", flush=True)
    test_model.train()
    for epoch in range(3):  # Just 3 epochs for testing
        total_loss = 0
        batch_count = 0
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = test_model(sequences)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count
        print(f"   Test epoch {epoch+1}: Loss = {avg_loss:.4f}", flush=True)
    
    print("✅ All diagnostic tests passed successfully!", flush=True)
    print("🚀 Ready to proceed with full experiment", flush=True)
    
    # Now start the actual experiment
    print("\n" + "="*80, flush=True)
    print("🎯 STARTING FULL DEFINITIVE EXPERIMENT", flush=True)
    print("="*80, flush=True)
    
    # Import the full experiment code by executing it
    exec(open('/Users/bard/Code/genomic-sparse-attention/phase5_definitive_sequential_training.py').read())
    
except Exception as e:
    print(f"❌ ERROR in diagnostic: {str(e)}", flush=True)
    print(f"❌ Traceback:", flush=True)
    traceback.print_exc()
    sys.exit(1)
