import pytest
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from personaflow.core.system import PersonaSystem
from personaflow.utils import PromptManager


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
            torch_dtype="auto",
        )
        return model, tokenizer

    @pytest.fixture
    def system(self):
        return PersonaSystem()

    @pytest.fixture
    def prompt_manager(self):
        manager = PromptManager()
        manager.add_template(
            "merchant",
            """You are ${name}, a ${occupation} known for ${personality}.
            Background: ${background}
            Inventory: ${inventory}

            Previous context: ${context}
            Human: ${user_input}
            ${name}:""",
        )

        manager.add_template(
            "guard",
            """You are ${name}, a ${occupation} who is ${personality}.
            Background: ${background}

            Previous context: ${context}
            Human: ${user_input}
            Guard ${name}:""",
        )

        manager.add_template(
            "innkeeper",
            """You are ${name}, an ${occupation} who is ${personality}.
            Background: ${background}

            Previous context: ${context}
            Human: ${user_input}
            Innkeeper ${name}:""",
        )
        return manager

    def generate_response(self, llm, prompt: str) -> str:
        model, tokenizer = llm
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs, max_length=2048, temperature=0.7, top_p=0.9, do_sample=True
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    def test_basic_character_interaction(self, llm, system, prompt_manager):
        # Create a test character
        system.create_character(
            name="Tom the Merchant",
            prompt=prompt_manager.get_prompt(
                "merchant",
                name="Tom",
                occupation="merchant",
                personality="fair prices and friendly service",
                background="Running the shop for 20 years",
                inventory="swords, shields, and armor",
                context="",
                user_input="",
            ),
            background={
                "occupation": "merchant",
                "personality": "fair prices and friendly service",
                "background": "Running the shop for 20 years",
                "inventory": "swords, shields, and armor",
            },
        )

        # Get character and context
        character = system.get_character("Tom the Merchant")
        context = character.get_context()

        # Create prompt and generate response
        formatted_prompt = prompt_manager.get_prompt(
            "merchant",
            **context["background"],
            name="Tom",
            context=str(context.get("memories", [])),
            user_input="How much for that sword?"
        )

        response = self.generate_response(llm, formatted_prompt)

        # Add interaction to memory
        character.add_memory(
            content={"user": "How much for that sword?", "response": response}
        )

        # Verify memory was added
        updated_context = character.get_context()
        assert len(updated_context["memories"]) == 1

    @pytest.mark.parametrize("prompt_type", ["friendly", "hostile"])
    def test_different_interaction_types(
        self, llm, system, prompt_manager, prompt_type
    ):
        # Create guard character
        system.create_character(
            name="John the Guard",
            prompt=prompt_manager.get_prompt(
                "guard",
                name="John",
                occupation="town guard",
                personality="professional and dutiful",
                background="Serving the town for 5 years",
                context="",
                user_input="",
            ),
            background={
                "occupation": "town guard",
                "personality": "professional and dutiful",
                "background": "Serving the town for 5 years",
            },
        )

        prompts = {
            "friendly": "Good evening guard, lovely weather we're having.",
            "hostile": "Get out of my way, guard!",
        }

        character = system.get_character("John the Guard")
        context = character.get_context()

        formatted_prompt = prompt_manager.get_prompt(
            "guard",
            **context["background"],
            name="John",
            context=str(context.get("memories", [])),
            user_input=prompts[prompt_type]
        )

        response = self.generate_response(llm, formatted_prompt)
        assert len(response) > 0

    def test_memory_influence(self, llm, system, prompt_manager):
        # Create innkeeper character
        system.create_character(
            name="Mary the Innkeeper",
            prompt=prompt_manager.get_prompt(
                "innkeeper",
                name="Mary",
                occupation="innkeeper",
                personality="observant and chatty",
                background="Running the Blue Dragon Inn",
                context="",
                user_input="",
            ),
            background={
                "occupation": "innkeeper",
                "personality": "observant and chatty",
                "background": "Running the Blue Dragon Inn",
            },
        )

        character = system.get_character("Mary the Innkeeper")

        # Add initial memory
        character.add_memory(
            content={
                "user": "Have you seen any interesting travelers lately?",
                "response": "Yes, a group of dwarven merchants stayed here last night.",
            }
        )

        # Test response with memory context
        context = character.get_context()
        formatted_prompt = prompt_manager.get_prompt(
            "innkeeper",
            **context["background"],
            name="Mary",
            context=str(context.get("memories", [])),
            user_input="Tell me more about those dwarven merchants."
        )

        response = self.generate_response(llm, formatted_prompt)
        assert len(response) > 0
