from llm_deploy.llm_calculator import LLMCalculator

def main():
    models = [
        "mixtral:8x7b-text-v0.1-q5_K_M",
        "deepseek-coder:6.7b-base-q5_K_M",
        "mistral-openorca:7b-q5_K_M"
    ]
    context = 8192

    calculator = LLMCalculator()

    for model in models:
        model_size, context_size, total_size = calculator.calculate(model, context)

        print("="*40)
        print(f"Model: {model}")
        print(f"Context Size: {context}")
        print(f"Model Size (GB): {model_size:.2f}")
        print(f"Context Size (GB): {context_size:.2f}")
        print(f"Total Size (GB): {total_size:.2f}")
        print("="*40)
        print()

if __name__ == "__main__":
    main()
