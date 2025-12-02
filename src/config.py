# === LABORATORY 1: The "Mechanism" Lab (Open Source / White Box) ===
# Controlled variable: Alignment Stage. 
# Constants: Base model (Llama 3.1 8B), Training Data (Tulu recipe).
OPEN_SOURCE_MODELS = {
    # 1. The Vulnerable Instruction Follower
    "tulu_sft": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    
    # 2. The Robust/Safe Model (DPO Aligned)
    "tulu_dpo": "allenai/Llama-3.1-Tulu-3-8B-DPO",
    
    # 3. (Optional) State of the Art Open Source
    "tulu_rlvr": "allenai/Llama-3.1-Tulu-3-8B" 
}

# === LABORATORY 2: The "Real World" Lab (Closed Source / Black Box) ===
# Purpose: Benchmark attack severity against industry standards.
CLOSED_SOURCE_MODELS = {
    "gpt_4o": "gpt-4o"
}

# Combine for iteration, or access individually
ALL_MODELS = {**OPEN_SOURCE_MODELS, **CLOSED_SOURCE_MODELS}

