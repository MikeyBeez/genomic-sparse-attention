#!/usr/bin/env python3
"""
Minimal debug test for data generation
"""
import random

# Test the basic motif planting logic
print("🧬 Debug test for genomic data generation...")

# Simplified motif structure
class SimpleMotif:
    def __init__(self, name, sequence, strength=0.5):
        self.name = name
        self.sequence = sequence
        self.strength = strength
        self.min_position = 10
        self.max_position = 50

# Create a simple motif
motifs = [
    SimpleMotif("TATA", "TATAAA", 0.8),
    SimpleMotif("CAAT", "CCAAT", 0.6)
]

def generate_simple_sequence(seq_length=100, num_motifs=1):
    """Generate a simple sequence with planted motifs."""
    # Create random background
    nucleotides = ['A', 'T', 'C', 'G']
    sequence = [random.choice(nucleotides) for _ in range(seq_length)]
    
    # Select motifs to plant
    selected_motifs = random.sample(motifs, min(num_motifs, len(motifs)))
    planted_motifs = []
    
    for motif in selected_motifs:
        # Find valid position
        max_pos = min(motif.max_position, seq_length - len(motif.sequence))
        min_pos = max(motif.min_position, 0)
        
        if min_pos < max_pos:
            position = random.randint(min_pos, max_pos)
            
            # Plant the motif
            for i, nucleotide in enumerate(motif.sequence):
                if position + i < len(sequence):  # Safety check
                    sequence[position + i] = nucleotide
            
            planted_motifs.append({
                'name': motif.name,
                'sequence': motif.sequence,
                'start': position,
                'end': position + len(motif.sequence),
                'strength': motif.strength
            })
    
    return ''.join(sequence), planted_motifs

# Test the simple generation
try:
    print("Testing simple sequence generation...")
    for i in range(5):
        sequence, planted = generate_simple_sequence(100, 2)
        print(f"Sample {i+1}: {len(sequence)} bp, {len(planted)} motifs")
        for motif in planted:
            print(f"  {motif['name']} at {motif['start']}-{motif['end']}: {motif['sequence']}")
    
    print("\n✅ Simple generation works!")
    
except Exception as e:
    import traceback
    print(f"❌ Error in simple generation: {e}")
    print(traceback.format_exc())

# Now test with torch tensors
print("\n=== Testing with PyTorch ===")
try:
    import torch
    print("✅ PyTorch available")
    
    # Test tensor creation
    sequences = []
    labels = []
    
    # Create mapping
    char_to_idx = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4, 'PAD': 5}
    
    for i in range(10):
        sequence_str, planted = generate_simple_sequence(100, 2)
        
        # Encode sequence
        encoded = [char_to_idx.get(char, 4) for char in sequence_str]  # N for unknown
        sequences.append(torch.tensor(encoded, dtype=torch.long))
        
        # Calculate strength
        strength = sum(m['strength'] for m in planted) / max(1, len(planted))
        labels.append(strength)
    
    # Stack tensors
    sequences_tensor = torch.stack(sequences)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    print(f"✅ Created tensors: sequences {sequences_tensor.shape}, labels {labels_tensor.shape}")
    print(f"Mean regulatory strength: {labels_tensor.mean():.3f}")
    
except Exception as e:
    import traceback
    print(f"❌ Error with PyTorch: {e}")
    print(traceback.format_exc())
