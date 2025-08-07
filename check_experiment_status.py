#!/usr/bin/env python3
"""
Monitor the 4-hour definitive sequential training experiment.
"""
import json
import time
from pathlib import Path
from datetime import datetime

def check_experiment_status():
    results_dir = Path('/Users/bard/Code/genomic-sparse-attention/results/phase5_definitive')
    results_file = results_dir / 'definitive_4hour_results.json'
    
    print("🔍 Checking 4-Hour Sequential Training Experiment Status")
    print("=" * 60)
    
    if results_file.exists():
        print("✅ EXPERIMENT COMPLETED!")
        
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            # Extract key metrics
            teacher_corr = results['success_metrics']['teacher_correlation']
            transfer_improvement = results['success_metrics']['transfer_improvement_percent']
            overall_success = results['success_metrics']['overall_success']
            total_hours = results['experiment_metadata']['total_time_hours']
            
            print(f"\n📊 FINAL RESULTS:")
            print(f"   Teacher Correlation: {teacher_corr:.3f}")
            print(f"   Transfer Improvement: {transfer_improvement:+.2f}%")
            print(f"   Overall Success: {'✅ YES' if overall_success else '❌ NO'}")
            print(f"   Total Time: {total_hours:.2f} hours")
            
            if overall_success:
                print(f"\n🎉 BREAKTHROUGH VALIDATED!")
                print(f"Sequential training (Joint → Sparse) definitively proven!")
            
        except Exception as e:
            print(f"❌ Error reading results: {e}")
            
    else:
        print("⏳ Experiment still running...")
        print(f"   Results will appear at: {results_file}")
        
        # Check if process is running
        import subprocess
        try:
            result = subprocess.run(['pgrep', '-f', 'phase5_definitive_sequential_training.py'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                print(f"   Active PIDs: {', '.join(pids)}")
                print("   🟢 Experiment is actively running")
            else:
                print("   🔴 No active experiment process found")
        except:
            print("   Status check unavailable")
    
    print(f"\n⏰ Current time: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    check_experiment_status()
