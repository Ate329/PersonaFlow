import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from personaflow.core import PersonaSystem
from personaflow.utils import PromptManager


def setup_model(use_gpu=False):
    model_name = "nvidia/Nemotron-Mini-4B-Instruct"

    # Determine device
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        device_map = None  # Will use single GPU
        print("Using GPU")
    else:
        device = "cpu"
        device_map = "auto"  # Will distribute across available devices
        print("Using CPU")

    # Create offload directory if needed
    offload_folder = "model_offload"
    os.makedirs(offload_folder, exist_ok=True)
    print(f"Using offload folder: {offload_folder}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device_map,
        offload_folder=offload_folder,
        low_cpu_mem_usage=True,
    )

    if device == "cuda":
        model = model.to(device)

    return model, tokenizer


def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_length=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main(use_gpu=False):
    # Initialize model and tokenizer
    model, tokenizer = setup_model(use_gpu)

    # Initialize PersonaSystem
    system = PersonaSystem()

    # Create prompt templates
    prompt_manager = PromptManager()

    # Add detective template
    prompt_manager.add_template(
        "detective",
        """You are ${name}, a brilliant detective known for ${traits}.
        Background: ${background}
        Current case: ${current_case}

        Previous context: ${context}
        Human: ${user_input}
        Detective ${name}:""",
    )

    # Add doctor template
    prompt_manager.add_template(
        "doctor",
        """You are ${name}, a medical doctor specialized in ${specialty}.
        Background: ${background}
        Current patient: ${current_patient}

        Previous context: ${context}
        Human: ${user_input}
        Dr. ${name}:""",
    )

    # Create characters
    system.create_character(
        name="Sherlock Holmes",
        prompt=prompt_manager.get_prompt(
            "detective",
            name="Sherlock Holmes",
            traits="exceptional deductive reasoning",
            background="Consulting detective from London",
            current_case="The mysterious emerald theft",
            context="",
            user_input="",
        ),
        background={
            "traits": "exceptional deductive reasoning",
            "background": "Consulting detective from London",
            "current_case": "The mysterious emerald theft",
        },
    )

    system.create_character(
        name="Dr. Watson",
        prompt=prompt_manager.get_prompt(
            "doctor",
            name="John Watson",
            specialty="military medicine",
            background="Army doctor and detective's assistant",
            current_patient="None at the moment",
            context="",
            user_input="",
        ),
        background={
            "specialty": "military medicine",
            "background": "Army doctor and detective's assistant",
            "current_patient": "None at the moment",
        },
    )

    # Main conversation loop
    current_character = "Sherlock Holmes"
    system.switch_active_character(current_character)

    print("\nCommands:")
    print("- Type 'switch' to change characters")
    print("- Type 'quit' to exit")
    print(f"\nCurrently speaking with: {current_character}")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "switch":
            # Toggle between characters
            current_character = (
                "Dr. Watson"
                if current_character == "Sherlock Holmes"
                else "Sherlock Holmes"
            )
            system.switch_active_character(current_character)
            print(f"\nSwitched to: {current_character}")
            continue

        # Get current character
        character = system.get_character(current_character)
        context = character.get_context(include_memories=True, memory_limit=5)

        # Format prompt based on character type
        template_name = "doctor" if current_character == "Dr. Watson" else "detective"
        formatted_prompt = prompt_manager.get_prompt(
            template_name,
            **context["background"],
            name=character.name,
            context=str(context.get("memories", [])),
            user_input=user_input,
        )

        # Generate and print response
        response = generate_response(model, tokenizer, formatted_prompt)
        print(f"\n{current_character}: {response}")

        # Add to character's memory
        character.add_memory(
            content={"user": user_input, "response": response},
            memory_type="interaction",
        )


if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("\nNVIDIA GPU detected!")
        use_gpu = input("Would you like to use GPU? (y/n): ").lower().startswith("y")
    else:
        print("\nNo NVIDIA GPU detected, using CPU")
        use_gpu = False

    try:
        main(use_gpu)
    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
    finally:
        # Clean up
        if os.path.exists("model_offload"):
            print("\nNOTE: The 'model_offload' folder contains temporary files.")
            print("You can safely delete it if you're not planning to reuse it.")
