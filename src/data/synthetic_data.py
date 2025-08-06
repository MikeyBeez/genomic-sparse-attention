"""
Synthetic Genomic Data Generation

Generates synthetic DNA sequences with planted regulatory motifs for validating
sparse attention mechanisms. Creates ground truth data where we know exactly
which positions should receive attention.
"""

import torch
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class RegulatoryMotif:
    """Represents a regulatory DNA motif."""
    name: str
    sequence: str
    strength: float  # Contribution to regulatory activity (0-1)
    min_position: int = 0  # Minimum position in sequence
    max_position: int = -1  # Maximum position (-1 = anywhere)


class SyntheticGenomicDataGenerator:
    """
    Generates synthetic genomic sequences with known regulatory elements.
    
    This allows us to validate that attention mechanisms correctly identify
    the planted regulatory motifs while ignoring the random "noise" sequence.
    """
    
    # Common regulatory motifs (simplified versions)
    REGULATORY_MOTIFS = [
        RegulatoryMotif("TATA_box", "TATAAA", 0.8, 50, 150),  # Promoter region
        RegulatoryMotif("CAAT_box", "CCAAT", 0.6, 100, 200),
        RegulatoryMotif("GC_box", "GGGCGG", 0.7, 80, 180),
        RegulatoryMotif("E_box", "CANNTG", 0.5, 200, 800),  # Enhancer
        RegulatoryMotif("AP1", "TGANTCA", 0.6, 300, 1000),
        RegulatoryMotif("NF_kB", "GGGACTTTCC", 0.9, 400, 1200),
        RegulatoryMotif("p53", "RRRCWWGYYY", 0.7, 500, 1500),
    ]
    
    def __init__(self, seq_length: int = 2000, vocab_size: int = 6):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # Nucleotide mapping: A=0, T=1, C=2, G=3, N=4, PAD=5
        self.nucleotides = ['A', 'T', 'C', 'G', 'N']
        self.char_to_idx = {char: idx for idx, char in enumerate(self.nucleotides + ['PAD'])}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        # Expand motifs with ambiguous nucleotides
        self.expanded_motifs = self._expand_motifs()
        
    def _expand_motifs(self) -> List[RegulatoryMotif]:
        """Expand motifs containing ambiguous nucleotides."""
        expanded = []
        
        ambiguous_codes = {
            'R': ['A', 'G'], 'Y': ['C', 'T'], 'W': ['A', 'T'], 'S': ['C', 'G'],
            'K': ['G', 'T'], 'M': ['A', 'C'], 'N': ['A', 'T', 'C', 'G']
        }
        
        for motif in self.REGULATORY_MOTIFS:
            if any(amb in motif.sequence for amb in ambiguous_codes.keys()):
                # Generate specific sequences from ambiguous motif
                sequences = [motif.sequence]
                for amb, nucs in ambiguous_codes.items():
                    new_sequences = []
                    for seq in sequences:
                        if amb in seq:
                            for nuc in nucs:
                                new_sequences.append(seq.replace(amb, nuc, 1))
                        else:
                            new_sequences.append(seq)
                    sequences = new_sequences
                
                # Create expanded motifs
                for i, seq in enumerate(sequences[:3]):  # Limit to avoid explosion
                    expanded.append(RegulatoryMotif(
                        f"{motif.name}_{i}", seq, motif.strength, 
                        motif.min_position, motif.max_position
                    ))
            else:
                expanded.append(motif)
        
        return expanded
    
    def generate_sequence_with_motifs(
        self, 
        num_motifs: int = 3,
        background_gc_content: float = 0.4
    ) -> Tuple[str, List[Dict], float]:
        """
        Generate a single sequence with planted regulatory motifs.
        
        Args:
            num_motifs: Number of regulatory motifs to plant
            background_gc_content: GC content for background sequence
            
        Returns:
            sequence: Generated DNA sequence
            planted_motifs: List of planted motif information
            regulatory_strength: Overall predicted regulatory strength
        """
        # Initialize with random background sequence
        sequence = self._generate_background_sequence(background_gc_content)
        sequence = list(sequence)
        
        # Select random motifs to plant
        if len(self.expanded_motifs) == 0 or num_motifs == 0:
            selected_motifs = []
        else:
            selected_motifs = random.sample(self.expanded_motifs, min(num_motifs, len(self.expanded_motifs)))
        planted_motifs = []
        total_strength = 0.0
        
        for motif in selected_motifs:
            # Find valid position range
            max_pos = motif.max_position if motif.max_position > 0 else self.seq_length - len(motif.sequence)
            min_pos = max(motif.min_position, 0)
            
            if min_pos < max_pos:
                # Plant the motif at random position within range
                position = random.randint(min_pos, max_pos - len(motif.sequence))
                
                # Check for overlaps with existing motifs
                overlap = False
                for existing in planted_motifs:
                    if (position < existing['end'] and position + len(motif.sequence) > existing['start']):
                        overlap = True
                        break
                
                if not overlap:
                    # Plant the motif
                    for i, nucleotide in enumerate(motif.sequence):
                        if nucleotide in self.nucleotides:
                            sequence[position + i] = nucleotide
                    
                    planted_motifs.append({
                        'name': motif.name,
                        'sequence': motif.sequence,
                        'start': position,
                        'end': position + len(motif.sequence),
                        'strength': motif.strength
                    })
                    total_strength += motif.strength
        
        # Calculate regulatory strength (normalized)
        regulatory_strength = min(1.0, total_strength / max(1, len(selected_motifs)))
        
        return ''.join(sequence), planted_motifs, regulatory_strength
    
    def _generate_background_sequence(self, gc_content: float) -> str:
        """Generate random background DNA sequence with specified GC content."""
        sequence = []
        
        for _ in range(self.seq_length):
            if random.random() < gc_content / 2:
                nucleotide = 'G'
            elif random.random() < gc_content / 2:
                nucleotide = 'C'
            elif random.random() < 0.5:
                nucleotide = 'A'
            else:
                nucleotide = 'T'
            sequence.append(nucleotide)
        
        return ''.join(sequence)
    
    def generate_dataset(
        self, 
        num_samples: int = 1000,
        positive_ratio: float = 0.6,
        num_motifs_range: Tuple[int, int] = (2, 5)
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Generate a complete dataset of synthetic genomic sequences.
        
        Args:
            num_samples: Total number of sequences to generate
            positive_ratio: Fraction of sequences with strong regulatory activity
            num_motifs_range: Range of number of motifs per sequence
            
        Returns:
            sequences: Encoded sequences [num_samples, seq_length]
            labels: Regulatory strength labels [num_samples]
            metadata: List of dictionaries with motif information
        """
        sequences = []
        labels = []
        metadata = []
        
        print(f"Generating {num_samples} synthetic genomic sequences...")
        
        for i in range(num_samples):
            # Determine if this should be a positive (regulatory) or negative sequence
            is_positive = random.random() < positive_ratio
            
            if is_positive:
                # Generate sequence with strong regulatory motifs
                num_motifs = random.randint(*num_motifs_range)
            else:
                # Generate sequence with weak/no regulatory motifs
                num_motifs = random.randint(0, 2)
            
            sequence_str, planted_motifs, strength = self.generate_sequence_with_motifs(num_motifs)
            
            # Encode sequence to integers
            encoded_sequence = self._encode_sequence(sequence_str)
            
            sequences.append(encoded_sequence)
            labels.append(strength)
            metadata.append({
                'index': i,
                'sequence_str': sequence_str,
                'planted_motifs': planted_motifs,
                'regulatory_strength': strength,
                'num_motifs': len(planted_motifs)
            })
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} sequences")
        
        # Convert to tensors
        sequences_tensor = torch.stack(sequences)
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        
        print(f"\nDataset generation complete!")
        print(f"Sequences shape: {sequences_tensor.shape}")
        print(f"Labels shape: {labels_tensor.shape}")
        print(f"Mean regulatory strength: {labels_tensor.mean():.3f}")
        print(f"Positive samples (>0.5): {(labels_tensor > 0.5).sum()}/{num_samples}")
        
        return sequences_tensor, labels_tensor, metadata
    
    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode DNA sequence string to tensor of integers."""
        encoded = []
        for char in sequence:
            if char in self.char_to_idx:
                encoded.append(self.char_to_idx[char])
            else:
                encoded.append(self.char_to_idx['N'])  # Unknown -> N
        
        # Pad or truncate to desired length
        if len(encoded) < self.seq_length:
            encoded.extend([self.char_to_idx['PAD']] * (self.seq_length - len(encoded)))
        elif len(encoded) > self.seq_length:
            encoded = encoded[:self.seq_length]
        
        return torch.tensor(encoded, dtype=torch.long)
    
    def create_attention_mask(self, metadata: List[Dict]) -> torch.Tensor:
        """
        Create ground truth attention mask showing where motifs are located.
        
        This provides the "ground truth" for where attention should focus.
        
        Args:
            metadata: Sequence metadata with motif positions
            
        Returns:
            attention_mask: [num_samples, seq_length] - 1 where motifs exist, 0 elsewhere
        """
        num_samples = len(metadata)
        attention_mask = torch.zeros(num_samples, self.seq_length)
        
        for i, sample_meta in enumerate(metadata):
            for motif in sample_meta['planted_motifs']:
                start, end = motif['start'], motif['end']
                attention_mask[i, start:end] = 1.0
        
        return attention_mask
    
    def visualize_sample(self, idx: int, metadata: List[Dict]) -> str:
        """Create a visualization of a sample sequence with motifs highlighted."""
        sample = metadata[idx]
        sequence = sample['sequence_str']
        motifs = sample['planted_motifs']
        
        # Create visualization string
        vis_lines = [f"Sample {idx}: Regulatory strength = {sample['regulatory_strength']:.3f}"]
        vis_lines.append("Sequence:")
        vis_lines.append(sequence)
        vis_lines.append("Motifs:   " + " " * len(sequence))
        
        # Mark motif positions
        motif_line = list(" " * len(sequence))
        for motif in motifs:
            for pos in range(motif['start'], motif['end']):
                if pos < len(motif_line):
                    motif_line[pos] = "*"
        vis_lines[-1] = "Motifs:   " + "".join(motif_line)
        
        # Add motif details
        vis_lines.append("\nPlanted motifs:")
        for motif in motifs:
            vis_lines.append(f"  {motif['name']}: {motif['sequence']} at pos {motif['start']}-{motif['end']} (strength: {motif['strength']:.2f})")
        
        return "\n".join(vis_lines)


# Dataset class for PyTorch DataLoader
class SyntheticGenomicDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapper for synthetic genomic data."""
    
    def __init__(self, sequences: torch.Tensor, labels: torch.Tensor, metadata: List[Dict]):
        self.sequences = sequences
        self.labels = labels
        self.metadata = metadata
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
    
    def get_metadata(self, idx):
        return self.metadata[idx]
    
    def get_attention_mask(self):
        """Get ground truth attention mask for all samples."""
        generator = SyntheticGenomicDataGenerator(seq_length=self.sequences.shape[1])
        return generator.create_attention_mask(self.metadata)
