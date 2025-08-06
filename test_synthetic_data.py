#!/usr/bin/env python3
"""
Simple test script for synthetic data generation
"""
import sys
import os

# Add src to path
sys.path.append('/Users/bard/Code/genomic-sparse-attention/src')

print("🧬 Testing synthetic data generation...")

try:
    from data.synthetic_data import SyntheticGenomicDataGenerator
    print("✅ Successfully imported SyntheticGenomicDataGenerator")
    
    # Create generator with smaller sequences for testing
    generator = SyntheticGenomicDataGenerator(seq_length=200)
    print(f"✅ Created generator with {len(generator.expanded_motifs)} expanded motifs")
    
    # List available motifs
    print("\nAvailable regulatory motifs:")
    for i, motif in enumerate(generator.expanded_motifs[:5]):  # Show first 5
        print(f"  {i+1}. {motif.name}: {motif.sequence} (strength: {motif.strength})")
    
    # Test single sequence generation
    print("\n=== Testing Single Sequence Generation ===")
    sequence, motifs, strength = generator.generate_sequence_with_motifs(num_motifs=2)
    print(f"✅ Generated sequence of length {len(sequence)}")
    print(f"✅ Planted {len(motifs)} motifs")
    print(f"✅ Regulatory strength: {strength:.3f}")
    
    # Show the sequence with motif locations
    print(f"\nFirst 100 nucleotides: {sequence[:100]}")
    print("Planted motifs:")
    for motif in motifs:
        print(f"  {motif['name']}: positions {motif['start']}-{motif['end']} = '{motif['sequence']}'")
        
    # Test small dataset generation
    print("\n=== Testing Dataset Generation ===")
    sequences, labels, metadata = generator.generate_dataset(num_samples=20)
    print(f"✅ Generated dataset: sequences {sequences.shape}, labels {labels.shape}")
    print(f"Mean regulatory strength: {labels.mean():.3f}")
    print(f"Positive samples (>0.5): {(labels > 0.5).sum()}/{len(labels)}")
    
    # Show sample metadata
    print("\nSample metadata:")
    for i in range(min(3, len(metadata))):
        sample = metadata[i]
        print(f"  Sample {i}: {len(sample['planted_motifs'])} motifs, strength: {sample['regulatory_strength']:.3f}")
    
    print("\n🎉 Synthetic data generation working correctly!")
    
except Exception as e:
    import traceback
    print(f"❌ Error: {e}")
    print("Traceback:")
    print(traceback.format_exc())
