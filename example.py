import gc
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from personaflow.core import PersonaSystem
from personaflow.utils import PromptManager


def clear_memory():
    """Clear memory and garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def setup_model(use_gpu=False):
    """
    Initialize the model with optimizations
    Args:
        use_gpu: Boolean to force GPU usage if available
    """
    model_name = "nvidia/Nemotron-Mini-4B-Instruct"

    # Determine device
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
        device_map = None
    else:
        device = "cpu"
        device_map = "auto"

    print(f"Using device: {device}")

    # Load tokenizer with optimizations
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, model_max_length=2048
    )

    # Create offload directory if needed
    offload_folder = "model_offload"
    os.makedirs(offload_folder, exist_ok=True)

    # Load model with CPU optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map=device_map,
        offload_folder=offload_folder,
        use_cache=True,
        low_cpu_mem_usage=True,
    )

    if device == "cuda":
        model = model.to(device)

    # Enable model optimizations
    model.config.use_cache = True

    # Warmup run
    print("Performing warmup inference...")
    _ = generate_response(model, tokenizer, "Hello")
    clear_memory()

    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=2048):
    """Generate a response with optimized parameters"""
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048, padding=False
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            min_length=32,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(
        outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return response


def generate_batch_responses(model, tokenizer, prompts, max_length=2048):
    """Generate responses for multiple prompts in batch"""
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_length=max_length, num_return_sequences=len(prompts)
        )

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


# Cache for prompt templates
cached_prompts = {}


def get_cached_prompt(prompt_manager, template_name, **kwargs):
    """Get or create cached prompt template"""
    cache_key = f"{template_name}_{hash(frozenset(kwargs.items()))}"
    if cache_key not in cached_prompts:
        cached_prompts[cache_key] = prompt_manager.get_prompt(template_name, **kwargs)
    return cached_prompts[cache_key]


def main(use_gpu=False):
    # Initialize the model and tokenizer
    model, tokenizer = setup_model(use_gpu)

    # Initialize the PersonaSystem
    system = PersonaSystem()

    # Create a PromptManager and add templates
    prompt_manager = PromptManager()
    detective_template = """You are ${name}, a brilliant detective known for ${traits}.
    Background: ${background}
    Current case: ${current_case}

    Previous context: ${context}

    Human: ${user_input}
    Detective ${name}:"""

    prompt_manager.add_template("detective", detective_template)

    # Create initial detective prompt
    detective_prompt = get_cached_prompt(
        prompt_manager,
        "detective",
        name="Sherlock Holmes",
        traits="exceptional deductive reasoning and keen observation",
        background="Consulting detective based in London, 221B Baker Street",
        current_case="The mysterious disappearance of Lord Blackwood's prized emerald",
        context="",
        user_input="",
    )

    # Create detective character
    detective = system.create_character(
        name="Sherlock Holmes",
        prompt=detective_prompt,
        background={
            "traits": "exceptional deductive reasoning and keen observation",
            "background": "Consulting detective based in London, 221B Baker Street",
            "current_case": "The mysterious disappearance of Lord Blackwood's prized emerald",
        },
        memory_config={
            "max_memories": 5,
            "summary_threshold": 3,
            "auto_summarize": True,
        },
    )

    # Simulate a conversation
    conversations = [
        "I found a green fabric thread at the crime scene. What do you make of it?",
        "The butler mentioned seeing a tall figure in the garden at midnight.",
        "Lord Blackwood's diary shows he was meeting someone in secret.",
    ]

    try:
        for user_input in conversations:
            # Get character context
            context = detective.get_context(include_memories=True, memory_limit=3)

            # Format the prompt using cache
            formatted_prompt = get_cached_prompt(
                prompt_manager,
                "detective",
                name=detective.name,
                traits=context["background"]["traits"],
                background=context["background"]["background"],
                current_case=context["background"]["current_case"],
                context=str(context.get("memories", [])),
                user_input=user_input,
            )

            # Generate response
            response = generate_response(model, tokenizer, formatted_prompt)

            # Add the interaction to memory
            detective.add_memory(
                content={"user": user_input, "response": response},
                memory_type="interaction",
            )

            # Print the conversation
            print(f"\nHuman: {user_input}")
            print(f"Sherlock: {response}\n")
            print("-" * 50)

            # Clear memory periodically
            clear_memory()

    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
    finally:
        # Final cleanup
        clear_memory()


if __name__ == "__main__":
    # For CPU-only systems
    main(use_gpu=False)

    # For NVIDIA GPU systems, uncomment the following line instead:
    # main(use_gpu=True)
