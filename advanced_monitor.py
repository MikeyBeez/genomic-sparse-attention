#!/usr/bin/env python3
"""
Advanced progress monitor for the 4-hour sequential training experiment.
"""
import os
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

def get_process_runtime(pid):
    """Get how long a process has been running."""
    try:
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'etime='], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        return "Unknown"
    except:
        return "Unknown"

def check_log_files():
    """Check for any log files or intermediate outputs."""
    results_dir = Path('/Users/bard/Code/genomic-sparse-attention/results/phase5_definitive')
    project_dir = Path('/Users/bard/Code/genomic-sparse-attention')
    
    log_files = []
    
    # Check for any log files
    for pattern in ['*.log', '*.txt', '*progress*', '*intermediate*']:
        log_files.extend(list(project_dir.glob(pattern)))
        if results_dir.exists():
            log_files.extend(list(results_dir.glob(pattern)))
    
    return log_files

def estimate_progress():
    """Estimate experiment progress based on time elapsed."""
    start_time_str = "10:58:36"  # From our launch
    start_time = datetime.strptime(start_time_str, "%H:%M:%S").replace(
        year=datetime.now().year,
        month=datetime.now().month, 
        day=datetime.now().day
    )
    
    current_time = datetime.now()
    elapsed = current_time - start_time
    elapsed_minutes = elapsed.total_seconds() / 60
    
    total_minutes = 240  # 4 hours
    progress_percent = (elapsed_minutes / total_minutes) * 100
    
    # Estimate current stage
    if elapsed_minutes < 60:
        current_stage = "Stage 1: Embedding Learning (500 epochs)"
        stage_progress = (elapsed_minutes / 60) * 100
    elif elapsed_minutes < 120:
        current_stage = "Stage 2: Task Learning (400 epochs)"  
        stage_progress = ((elapsed_minutes - 60) / 60) * 100
    elif elapsed_minutes < 125:
        current_stage = "Stage 3: Knowledge Extraction"
        stage_progress = ((elapsed_minutes - 120) / 5) * 100
    elif elapsed_minutes < 185:
        current_stage = "Stage 4a: Guided Student Training (300 epochs)"
        stage_progress = ((elapsed_minutes - 125) / 60) * 100
    elif elapsed_minutes < 245:
        current_stage = "Stage 4b: Baseline Student Training (300 epochs)"
        stage_progress = ((elapsed_minutes - 185) / 60) * 100
    else:
        current_stage = "Final Evaluation & Analysis"
        stage_progress = 100
    
    return {
        'elapsed_minutes': elapsed_minutes,
        'progress_percent': min(progress_percent, 100),
        'current_stage': current_stage,
        'stage_progress': min(stage_progress, 100),
        'estimated_completion': start_time + timedelta(hours=4)
    }

def main():
    print("🔍 Advanced 4-Hour Sequential Training Experiment Monitor")
    print("=" * 70)
    
    # Check if experiment is running
    try:
        result = subprocess.run(['pgrep', '-f', 'phase5_definitive_sequential_training.py'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            print(f"✅ Experiment RUNNING - PIDs: {', '.join(pids)}")
            
            # Get runtime for main process
            if pids:
                runtime = get_process_runtime(pids[0])
                print(f"📊 Runtime: {runtime}")
        else:
            print("❌ Experiment NOT RUNNING")
            return
    except:
        print("⚠️  Cannot check process status")
    
    # Estimate progress
    progress = estimate_progress()
    
    print(f"\n⏱️  PROGRESS ESTIMATION:")
    print(f"   Elapsed: {progress['elapsed_minutes']:.1f} minutes")
    print(f"   Overall Progress: {progress['progress_percent']:.1f}%")
    print(f"   Current Stage: {progress['current_stage']}")
    print(f"   Stage Progress: {progress['stage_progress']:.1f}%")
    print(f"   Expected Completion: {progress['estimated_completion'].strftime('%H:%M:%S')}")
    
    # Progress bar
    bar_length = 50
    filled_length = int(bar_length * progress['progress_percent'] / 100)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f"   [{bar}] {progress['progress_percent']:.1f}%")
    
    # Check for log files
    log_files = check_log_files()
    if log_files:
        print(f"\n📄 Found {len(log_files)} potential log files:")
        for log_file in log_files[:5]:  # Show first 5
            print(f"   {log_file}")
    else:
        print(f"\n📄 No intermediate log files found")
    
    # Check results directory
    results_dir = Path('/Users/bard/Code/genomic-sparse-attention/results/phase5_definitive')
    if results_dir.exists():
        files = list(results_dir.iterdir())
        if files:
            print(f"\n📁 Results directory contents:")
            for file in files:
                print(f"   {file.name}")
        else:
            print(f"\n📁 Results directory exists but empty")
    else:
        print(f"\n📁 Results directory not yet created")
    
    # Memory and CPU check (if available)
    try:
        # Check system load
        load_result = subprocess.run(['uptime'], capture_output=True, text=True)
        if load_result.returncode == 0:
            print(f"\n🖥️  System Load: {load_result.stdout.strip().split('load average:')[1]}")
    except:
        pass
    
    print(f"\n⏰ Monitor run at: {datetime.now().strftime('%H:%M:%S')}")
    
    # Provide next check recommendation
    if progress['progress_percent'] < 90:
        next_check = 30 if progress['progress_percent'] < 50 else 15
        print(f"💡 Recommend checking again in {next_check} minutes")

if __name__ == "__main__":
    main()
