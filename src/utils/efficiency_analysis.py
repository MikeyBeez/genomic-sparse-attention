"""
Utilities for computational efficiency measurement and visualization.
"""

import time
import torch
import psutil
import numpy as np
from typing import Dict, List, Callable, Any
import matplotlib.pyplot as plt


class ComputationalProfiler:
    """Profile computational efficiency of models and operations."""
    
    def __init__(self):
        self.profiles = {}
    
    def profile_model(
        self, 
        model: torch.nn.Module, 
        input_data: torch.Tensor, 
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        Profile a model's computational efficiency.
        
        Args:
            model: PyTorch model to profile
            input_data: Sample input tensor
            num_runs: Number of timing runs
            warmup_runs: Number of warmup runs (not counted)
            
        Returns:
            Dictionary with timing and memory statistics
        """
        model.eval()
        device = next(model.parameters()).device
        input_data = input_data.to(device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_data)
        
        # Memory before
        if device.type == 'cuda':
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated()
        else:
            memory_before = psutil.Process().memory_info().rss
        
        # Timing runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                output = model(input_data)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        # Memory after
        if device.type == 'cuda':
            memory_after = torch.cuda.memory_allocated()
            memory_unit = 'bytes'
        else:
            memory_after = psutil.Process().memory_info().rss
            memory_unit = 'bytes'
        
        # Calculate statistics
        times = np.array(times)
        
        return {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'memory_usage': memory_after - memory_before,
            'memory_unit': memory_unit,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)  # Assuming float32
        }
    
    def compare_models(
        self,
        models: Dict[str, torch.nn.Module],
        input_data: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """Compare computational efficiency of multiple models."""
        
        results = {}
        
        for name, model in models.items():
            print(f"Profiling {name}...")
            results[name] = self.profile_model(model, input_data, num_runs)
        
        return results
    
    def visualize_comparison(
        self,
        comparison_results: Dict[str, Dict[str, float]],
        save_path: str = None
    ):
        """Visualize model comparison results."""
        
        model_names = list(comparison_results.keys())
        metrics = ['mean_time', 'memory_usage', 'num_parameters', 'model_size_mb']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [comparison_results[name][metric] for name in model_names]
            
            bars = axes[i].bar(model_names, values, alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if metric == 'mean_time':
                    label = f'{value:.4f}s'
                elif metric in ['memory_usage', 'model_size_mb']:
                    label = f'{value:.2f}'
                else:
                    label = f'{value:,}'
                
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           label, ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def calculate_efficiency_gains(
        self,
        baseline_results: Dict[str, float],
        improved_results: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate efficiency gains compared to baseline."""
        
        gains = {}
        
        # Speed improvement
        if baseline_results['mean_time'] > 0:
            gains['speedup'] = baseline_results['mean_time'] / improved_results['mean_time']
        
        # Memory reduction
        gains['memory_reduction_ratio'] = baseline_results['memory_usage'] / improved_results['memory_usage']
        gains['memory_savings_percent'] = (1 - improved_results['memory_usage'] / baseline_results['memory_usage']) * 100
        
        # Parameter reduction
        gains['parameter_reduction_ratio'] = baseline_results['num_parameters'] / improved_results['num_parameters']
        gains['parameter_savings_percent'] = (1 - improved_results['num_parameters'] / baseline_results['num_parameters']) * 100
        
        # Model size reduction
        gains['size_reduction_ratio'] = baseline_results['model_size_mb'] / improved_results['model_size_mb']
        gains['size_savings_percent'] = (1 - improved_results['model_size_mb'] / baseline_results['model_size_mb']) * 100
        
        return gains


class AttentionAnalyzer:
    """Analyze and visualize attention patterns."""
    
    @staticmethod
    def calculate_attention_sparsity(attention_weights: torch.Tensor, threshold: float = 0.1) -> float:
        """
        Calculate the sparsity of attention weights.
        
        Args:
            attention_weights: Attention weights tensor
            threshold: Threshold below which weights are considered sparse
            
        Returns:
            Sparsity ratio (0 = dense, 1 = completely sparse)
        """
        total_weights = attention_weights.numel()
        sparse_weights = (attention_weights < threshold).sum().item()
        return sparse_weights / total_weights
    
    @staticmethod
    def calculate_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
        """Calculate entropy of attention distributions."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        attention_weights = attention_weights + epsilon
        
        # Normalize to ensure it's a proper distribution
        attention_weights = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
        
        # Calculate entropy
        entropy = -torch.sum(attention_weights * torch.log(attention_weights), dim=-1)
        return entropy
    
    @staticmethod
    def visualize_attention_patterns(
        attention_weights: torch.Tensor,
        sequence_metadata: Dict = None,
        save_path: str = None,
        title: str = "Attention Patterns"
    ):
        """Visualize attention weight patterns."""
        
        # Convert to numpy if needed
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
        
        # Average across batch and heads if needed
        if attention_weights.ndim == 4:  # [batch, heads, seq, seq]
            attention_weights = attention_weights.mean(axis=(0, 1))
        elif attention_weights.ndim == 3:  # [heads, seq, seq] or [batch, seq, seq]
            attention_weights = attention_weights.mean(axis=0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Heatmap of full attention matrix
        im1 = ax1.imshow(attention_weights, cmap='Blues', aspect='auto')
        ax1.set_title(f'{title} - Full Matrix')
        ax1.set_xlabel('Key Position')
        ax1.set_ylabel('Query Position')
        plt.colorbar(im1, ax=ax1)
        
        # Attention focus (sum over keys for each query)
        attention_focus = attention_weights.sum(axis=1)
        ax2.plot(attention_focus, 'b-', linewidth=2)
        ax2.set_title(f'{title} - Attention Focus')
        ax2.set_xlabel('Query Position')
        ax2.set_ylabel('Total Attention')
        ax2.grid(True, alpha=0.3)
        
        # Add motif markers if metadata provided
        if sequence_metadata and 'planted_motifs' in sequence_metadata:
            for motif in sequence_metadata['planted_motifs']:
                ax2.axvspan(motif['start'], motif['end'], alpha=0.3, 
                           color='red', label=f"{motif['name']}")
            ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def compare_attention_patterns(
        patterns: Dict[str, torch.Tensor],
        save_path: str = None
    ):
        """Compare multiple attention patterns side by side."""
        
        num_patterns = len(patterns)
        fig, axes = plt.subplots(2, num_patterns, figsize=(5 * num_patterns, 10))
        
        if num_patterns == 1:
            axes = axes.reshape(2, 1)
        
        for i, (name, pattern) in enumerate(patterns.items()):
            if isinstance(pattern, torch.Tensor):
                pattern = pattern.detach().cpu().numpy()
            
            # Average if needed
            if pattern.ndim > 2:
                pattern = pattern.mean(axis=tuple(range(pattern.ndim - 2)))
            
            # Heatmap
            im = axes[0, i].imshow(pattern, cmap='Blues', aspect='auto')
            axes[0, i].set_title(f'{name} - Full Matrix')
            axes[0, i].set_xlabel('Key Position')
            axes[0, i].set_ylabel('Query Position')
            plt.colorbar(im, ax=axes[0, i])
            
            # Focus plot
            focus = pattern.sum(axis=1)
            axes[1, i].plot(focus, 'b-', linewidth=2)
            axes[1, i].set_title(f'{name} - Focus')
            axes[1, i].set_xlabel('Query Position')
            axes[1, i].set_ylabel('Total Attention')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def measure_computational_savings(
    full_model: torch.nn.Module,
    sparse_model: torch.nn.Module,
    test_input: torch.Tensor,
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Measure computational savings between full and sparse models.
    
    Args:
        full_model: Full attention model
        sparse_model: Sparse attention model
        test_input: Test input tensor
        num_runs: Number of timing runs
        
    Returns:
        Dictionary with savings metrics
    """
    profiler = ComputationalProfiler()
    
    # Profile both models
    full_profile = profiler.profile_model(full_model, test_input, num_runs)
    sparse_profile = profiler.profile_model(sparse_model, test_input, num_runs)
    
    # Calculate savings
    savings = profiler.calculate_efficiency_gains(full_profile, sparse_profile)
    
    print("=== Computational Savings Analysis ===")
    print(f"Speedup: {savings['speedup']:.2f}x")
    print(f"Memory savings: {savings['memory_savings_percent']:.1f}%")
    print(f"Parameter reduction: {savings['parameter_savings_percent']:.1f}%")
    print(f"Model size reduction: {savings['size_savings_percent']:.1f}%")
    
    return {
        'full_profile': full_profile,
        'sparse_profile': sparse_profile,
        'savings': savings
    }
