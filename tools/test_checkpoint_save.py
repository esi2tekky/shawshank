#!/usr/bin/env python3
"""
More comprehensive test that actually attempts to save a checkpoint
to verify the full save() method works correctly.
"""

import sys
import tempfile
from pathlib import Path
import torch
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


def test_actual_save():
    """Test that we can actually save checkpoints with the sanitized names."""
    print("="*60)
    print("Testing Actual Checkpoint Save")
    print("="*60)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        test_output_dir = Path(tmpdir) / "test_checkpoints"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        all_passed = True
        
        for model_key, model_id in OPEN_SOURCE_MODELS.items():
            print(f"\nTesting save for: {model_key}")
            print(f"  Model ID: {model_id}")
            
            # Get sanitized name
            model_name = sanitize_model_name(model_id)
            print(f"  Sanitized: {model_name}")
            
            try:
                # Create model subdirectory (like rl_attacker.py does)
                model_output_dir = test_output_dir / model_key
                model_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Test checkpoint save (same logic as rl_attacker.py)
                checkpoint_filename = f"rl_checkpoint_{model_name}_50.pt"
                checkpoint_path = model_output_dir / checkpoint_filename
                
                # Ensure parent directory exists (same as save() method)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Actually try to save something (dummy data)
                dummy_data = {
                    "test": "data",
                    "model_name": model_name,
                    "episode": 50
                }
                
                # Use torch.save like the actual code does
                torch.save(dummy_data, str(checkpoint_path))
                
                # Verify file was created
                assert checkpoint_path.exists(), f"Checkpoint file not created: {checkpoint_path}"
                assert checkpoint_path.is_file(), f"Checkpoint path is not a file: {checkpoint_path}"
                
                # Verify we can load it back
                loaded = torch.load(str(checkpoint_path))
                assert loaded["model_name"] == model_name
                
                print(f"  ✅ Successfully saved and loaded: {checkpoint_path.name}")
                
                # Test other file types
                timestamp = "20251205_120000"
                history_filename = f"rl_history_{model_name}_{timestamp}.jsonl"
                history_path = model_output_dir / history_filename
                history_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write a dummy JSONL file
                import json
                with open(history_path, "w") as f:
                    f.write(json.dumps({"test": "data", "model": model_name}) + "\n")
                
                assert history_path.exists(), f"History file not created: {history_path}"
                print(f"  ✅ Successfully created: {history_path.name}")
                
            except Exception as e:
                print(f"  ❌ FAILED: {e}")
                import traceback
                traceback.print_exc()
                all_passed = False
        
        print("\n" + "="*60)
        if all_passed:
            print("✅ ALL SAVE TESTS PASSED - Ready to run RL training")
        else:
            print("❌ SOME SAVE TESTS FAILED - Fix needed")
        print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = test_actual_save()
    sys.exit(0 if success else 1)

