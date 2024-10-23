from personaflow.core.system import PersonaSystem
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class RPGGame:
    def __init__(self):
        # Initialize the system
        self.system = PersonaSystem()
        
        # Initialize LLM
        model_name = "microsoft/Phi-3.5-mini-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create offload directory if it doesn't exist
        offload_folder = "model_offload"
        os.makedirs(offload_folder, exist_ok=True)
        
        # Load model with proper configuration
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            offload_folder=offload_folder,
            low_cpu_mem_usage=True,
            torch_dtype="auto"
        )
        
        # Initialize characters
        self._setup_characters()

    def _setup_characters(self):
        # Create a merchant
        self.system.create_character(
            name="merchant",
            prompt="""You are Tom, a friendly merchant in a medieval fantasy town.
                     You sell weapons and armor. You are known for fair prices and honest deals.
                     You've been running your shop for 20 years and know all the local gossip.""",
            background={
                "name": "Tom",
                "occupation": "Merchant",
                "personality": "Friendly and helpful",
                "inventory": ["swords", "shields", "armor", "potions"],
                "location": "Market District"
            }
        )

        # Create a guard
        self.system.create_character(
            name="guard",
            prompt="""You are John, a town guard in a medieval fantasy town.
                     You maintain order and protect citizens. You take your duty seriously
                     but are fair in your dealings with people.""",
            background={
                "name": "John",
                "occupation": "Guard",
                "personality": "Professional and dutiful",
                "patrol_area": "Market District and Town Square"
            }
        )

    def generate_response(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1,
            use_cache=True
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def interact_with_character(self, character_name: str, user_input: str):
        # Get character
        character = self.system.get_character(character_name)
        
        # Get character context
        context = character.get_context()
        
        # Create prompt
        prompt = f"""<s>[INST] You are {context['background']['name']}. Here is your context:
{context}

A person says to you: "{user_input}"

Respond in character, considering your background, personality, and previous interactions.
[/INST]"""

        # Generate response
        response = self.generate_response(prompt)

        # Store interaction in character's memory
        self.system.add_interaction(
            character_name=character_name,
            content={
                "user": user_input,
                "response": response
            }
        )

        return response

    def broadcast_event(self, event_description: str):
        """Broadcast an event to all characters"""
        self.system.broadcast_interaction(
            content={
                "event": event_description,
                "type": "town_event"
            }
        )

    def save_game(self, filename: str):
        """Save current game state"""
        import json
        with open(filename, 'w') as f:
            json.dump(self.system.to_dict(), f, indent=2)

    def load_game(self, filename: str):
        """Load game state"""
        import json
        with open(filename, 'r') as f:
            data = json.load(f)
            self.system = PersonaSystem.from_dict(data)

def main():
    # Initialize game
    game = RPGGame()

    print("Welcome to the Medieval Fantasy Town!")
    print("Available characters: merchant, guard")
    print("Type 'quit' to exit")

    while True:
        # Get character selection
        character = input("\nWho would you like to talk to? ").lower()
        if character == 'quit':
            break

        if character not in ['merchant', 'guard']:
            print("Invalid character. Please choose 'merchant' or 'guard'")
            continue

        # Get user input
        user_input = input(f"\nWhat would you like to say to the {character}? ")
        if user_input.lower() == 'quit':
            break

        # Get response
        response = game.interact_with_character(character, user_input)
        print(f"\n{character.title()}: {response}")

        # Occasionally broadcast events
        if "dragon" in user_input.lower():
            game.broadcast_event("A dragon was spotted flying over the town!")
            print("\n[Event broadcasted to all characters]")

    # Save game state
    game.save_game("game_state.json")
    print("\nGame state saved!")

if __name__ == "__main__":
    main()