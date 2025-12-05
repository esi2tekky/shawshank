from src.target.target_factory import load_target

def test_models():
    # 1. Test OpenAI
    print("--- Testing OpenAI ---")
    gpt = load_target("gpt-3.5-turbo")
    print(gpt.query("Say 'Hello OpenAI'"))

    # 2. Test Local (Tulu 3)
    # Ensure you have GPU memory for this step
    print("\n--- Testing vLLM ---")
    local = load_target("allenai/Llama-3.1-Tulu-3-8B-SFT")
    print(local.query("Say 'Hello Local'"))

if __name__ == "__main__":
    test_models()

