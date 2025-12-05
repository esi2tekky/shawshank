import argparse
from src.config import ALL_MODELS, OPEN_SOURCE_MODELS
from src.target.target_factory import load_target

def run_comparative_scan(target_keys):
    """
    Iterates through selected models to prepare for attacks.
    """
    for key in target_keys:
        model_id = ALL_MODELS.get(key)
        if not model_id:
            print(f"Error: {key} not found in config.")
            continue
            
        print(f"\n==========================================")
        print(f"INITIALIZING TARGET: {key}")
        print(f"ID: {model_id}")
        print(f"==========================================")
        
        # In Phase 3, the Attacker loop goes here.
        # For Phase 2, we just load to verify memory/auth.
        try:
            target = load_target(model_id)
            # Dry run query
            response = target.query("Test probe.")
            print(f"Status: ONLINE | Response len: {len(response['text'])}")
            
            # CRITICAL: Clean up GPU memory for the next model
            if hasattr(target, 'llm'):
                import gc
                import torch
                del target.llm
                gc.collect()
                torch.cuda.empty_cache()
                print("GPU Memory cleared.")
                
        except Exception as e:
            print(f"FAILED to load {key}: {e}")

if __name__ == "__main__":
    # Default to scanning the Mechanism Lab (Tulu models)
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["tulu_sft", "tulu_dpo"])
    args = parser.parse_args()
    
    run_comparative_scan(args.targets)

