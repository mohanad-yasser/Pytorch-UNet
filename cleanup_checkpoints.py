import os
from pathlib import Path
import shutil

def cleanup_checkpoints(keep_last_n=5):
    checkpoint_dir = Path('./checkpoints')
    if not checkpoint_dir.exists():
        return
    
    # Get all checkpoint files
    checkpoint_files = sorted(
        [f for f in checkpoint_dir.glob('checkpoint_epoch*.pth')],
        key=lambda x: int(x.stem.split('_')[-1].replace('epoch', ''))
    )
    
    # Keep only the last N checkpoints
    files_to_remove = checkpoint_files[:-keep_last_n]
    
    # Create backup directory
    backup_dir = checkpoint_dir / 'old_checkpoints'
    backup_dir.mkdir(exist_ok=True)
    
    # Move old checkpoints to backup
    for file in files_to_remove:
        try:
            shutil.move(str(file), str(backup_dir / file.name))
            print(f"Moved {file.name} to backup")
        except Exception as e:
            print(f"Error moving {file.name}: {e}")

if __name__ == '__main__':
    cleanup_checkpoints(keep_last_n=5) 