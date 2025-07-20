from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LLMChatbot:
    def __init__(self, model_name="microsoft/DialoGPT-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def chat(self, prompt, history=None):
        # DialoGPT expects each turn as a separate input
        input_ids = None
        for msg in (history or []):
            if input_ids is None:
                input_ids = self.tokenizer.encode(msg["content"] + self.tokenizer.eos_token, return_tensors="pt")
            else:
                new_input = self.tokenizer.encode(msg["content"] + self.tokenizer.eos_token, return_tensors="pt")
                input_ids = torch.cat([input_ids, new_input], dim=-1)
        # Add current prompt
        new_input = self.tokenizer.encode(prompt + self.tokenizer.eos_token, return_tensors="pt")
        input_ids = torch.cat([input_ids, new_input], dim=-1) if input_ids is not None else new_input
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.shape[1]+64,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()

def main():
    print("Welcome to the Hugging Face LLM Chatbot! Type 'exit' to quit.")
    chatbot = LLMChatbot()
    history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        reply = chatbot.chat(user_input, history)
        print(f"Bot: {reply}")
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()
