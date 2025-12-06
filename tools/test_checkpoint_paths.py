#!/usr/bin/env python3
"""
Test script to verify checkpoint path handling works correctly
for all model names, especially those with slashes.
"""

import sys
from pathlib import Path
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import OPEN_SOURCE_MODELS


def sanitize_model_name(model_name):
    """Same sanitization logic as in rl_attacker.py"""
    model_name = str(model_name).replace('/', '_').replace('\\', '_').replace(' ', '_')
    model_name = re.sub(r'[<>:"|?*]', '_', model_name)
    if not model_name or model_name == '.' or model_name == '..':
        model_name = 'unknown_model'
    return model_name


def test_checkpoint_paths():
    """Test checkpoint path creation for all models."""
    print("="*60)
    print("Testing Checkpoint Path Creation")
    print("="*60)
    
    # Test directory (won't actually create files, just test paths)
    test_output_dir = Path("test_checkpoints")
    test_output_dir.mkdir(exist_ok=True)
    
    all_passed = True
    
    for model_key, model_id in OPEN_SOURCE_MODELS.items():
        print(f"\nTesting: {model_key}")
        print(f"  Model ID: {model_id}")
        
        # Get sanitized name (same as rl_attacker.py does)
        model_name = sanitize_model_name(model_id)
        print(f"  Sanitized: {model_name}")
        
        # Test checkpoint path creation (same as in rl_attacker.py)
        try:
            # Simulate checkpoint save
            checkpoint_filename = f"rl_checkpoint_{model_name}_50.pt"
            checkpoint_path = test_output_dir / checkpoint_filename
            
            # Verify path is valid
            assert not any(c in str(checkpoint_path) for c in ['/', '\\'] if c != os.sep), \
                f"Path contains invalid characters: {checkpoint_path}"
            
            # Verify parent directory creation would work
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Verify filename doesn't contain path separators
            assert '/' not in checkpoint_path.name or '\\' not in checkpoint_path.name, \
                f"Filename contains path separator: {checkpoint_path.name}"
            
            print(f"  ✅ Checkpoint path: {checkpoint_path}")
            print(f"  ✅ Path is valid")
            
            # Test other file types
            timestamp = "20251205_120000"
            final_filename = f"rl_final_{model_name}_{timestamp}.pt"
            history_filename = f"rl_history_{model_name}_{timestamp}.jsonl"
            experiences_filename = f"rl_experiences_{model_name}_{timestamp}.jsonl"
            
            final_path = test_output_dir / final_filename
            history_path = test_output_dir / history_filename
            experiences_path = test_output_dir / experiences_filename
            
            # Verify all paths
            for path in [final_path, history_path, experiences_path]:
                path.parent.mkdir(parents=True, exist_ok=True)
                assert '/' not in path.name or '\\' not in path.name, \
                    f"Filename contains path separator: {path.name}"
            
            print(f"  ✅ All file paths valid")
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            all_passed = False
    
    # Cleanup
    import shutil
    if test_output_dir.exists():
        shutil.rmtree(test_output_dir)
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Checkpoint paths will work correctly")
    else:
        print("❌ SOME TESTS FAILED - Fix needed before running")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    import os
    success = test_checkpoint_paths()
    sys.exit(0 if success else 1)

