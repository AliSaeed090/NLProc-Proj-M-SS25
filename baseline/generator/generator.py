import os
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
load_dotenv()

class Generator:
    def __init__(self, model_name: str = "llama3.1-8b"):
        """
        Initialize Cerebras client and set model.
        """
        api_key = os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            raise ValueError("❌ CEREBRAS_API_KEY environment variable not set.")
        
        self.client = Cerebras(api_key=api_key)
        self.model_name = model_name

    def build_prompt(self, context: list, metadata: list, question: str, history: list = None) -> list:
        """
        Build the prompt messages in OpenAI-compatible format.
        """
        combined_context = "\n".join(context)
        meta_info = "\n".join(metadata)

        messages = []
        if history:
            for q, a in history[-3:]:
                messages.append({"role": "user", "content": q})
                messages.append({"role": "assistant", "content": a})

        # Add current context + question
        user_message = (
            "You are an expert assistant. Only answer if relevant information is found in the provided context. "
            "If unsure, say: 'I don’t know based on the provided context.'\n\n"
            f"Context Source Info:\n{meta_info}\n"
            f"Context:\n{combined_context}\n\n"
            f"Question:\n{question}"
        )
        messages.append({"role": "user", "content": user_message})
        return messages

    def generate_answer(self, messages: list, max_tokens: int = 1024) -> str:
        """
        Call Cerebras chat completion API.
        """
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            max_tokens=max_tokens
        )
        # Extract the assistant's reply
        print(response)

        return response.choices[0].message.content
