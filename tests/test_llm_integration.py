import pytest
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from personaflow.core.system import PersonaSystem

class TestLLMIntegration:
    @pytest.fixture(scope="module")
    def llm(self):
        model_name = "nvidia/Nemotron-Mini-4B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create offload directory if it doesn't exist
        offload_folder = "model_offload"
        os.makedirs(offload_folder, exist_ok=True)

        # Load model with proper offloading configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            offload_folder=offload_folder,
            low_cpu_mem_usage=True,
            torch_dtype="auto"
        )
        return model, tokenizer

    @pytest.fixture
    def system(self):
        return PersonaSystem()

    def generate_response(self, llm, prompt: str) -> str:
        model, tokenizer = llm

        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate with optimized settings for NPU
        outputs = model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            use_cache=True,
            do_sample=True
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def test_basic_character_interaction(self, llm, system):
        # Create a test character
        character = system.create_character(
            name="merchant",
            prompt="""You are a friendly merchant in a medieval fantasy town.
                     You sell weapons and armor. You are known for fair prices.""",
            background={
                "name": "Tom",
                "occupation": "Merchant",
                "personality": "Friendly and helpful",
                "inventory": ["swords", "shields", "armor"]
            }
        )

        # Get character context
        context = character.get_context()

        # Create full prompt with proper instruction format
        prompt = f"""<s>[INST] You are a merchant named Tom. Here is your context:
{context}

A customer asks: "How much for that sword?"

Respond in character, considering your background and personality.
[/INST]"""

        # Generate response
        response = self.generate_response(llm, prompt)

        # Add interaction to character memory
        system.add_interaction(
            character_name="merchant",
            content={
                "user": "How much for that sword?",
                "response": response
            }
        )

        # Verify memory was added
        updated_context = character.get_context()
        assert len(updated_context["memories"]) == 1

    @pytest.mark.parametrize("prompt_type", ["friendly", "hostile"])
    def test_different_interaction_types(self, llm, system, prompt_type):
        # Test how character responds to different types of interactions
        character = system.create_character(
            name="guard",
            prompt="You are a town guard in a medieval fantasy town.",
            background={
                "name": "John",
                "occupation": "Guard",
                "personality": "Professional and dutiful"
            }
        )

        prompts = {
            "friendly": "Good evening guard, lovely weather we're having.",
            "hostile": "Get out of my way, guard!"
        }

        context = character.get_context()
        prompt = f"""<s>[INST] Based on this context:
{context}

Respond to: "{prompts[prompt_type]}"
[/INST]"""

        response = self.generate_response(llm, prompt)
        assert len(response) > 0

    def test_memory_influence(self, llm, system):
        # Test how previous interactions influence responses
        character = system.create_character(
            name="innkeeper",
            prompt="You are an innkeeper in a medieval fantasy town.",
            background={
                "name": "Mary",
                "occupation": "Innkeeper",
                "personality": "Observant and chatty"
            }
        )

        # Add some initial memories
        system.add_interaction(
            character_name="innkeeper",
            content={
                "user": "Have you seen any interesting travelers lately?",
                "response": "Yes, a group of dwarven merchants stayed here last night."
            }
        )

        context = character.get_context()
        prompt = f"""<s>[INST] Based on this context:
{context}

Respond to: "Tell me more about those dwarven merchants."
[/INST]"""

        response = self.generate_response(llm, prompt)
        assert len(response) > 0
