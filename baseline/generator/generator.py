from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class Generator:
    def __init__(self, model_name: str = "google/flan-t5-large", device: str = None):
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
    def build_prompt(self, context: list, question: str) -> str:
        """
        Format a prompt by joining context chunks and appending the question.
        """
        combined_context = "\n".join(context)
        prompt = (
            "You are an expert assistant that only answers questions about countries if the relevant information is available in the given context.\n"
            "Answer strictly based on the context provided. Only mention countries that are explicitly asked about in the question.\n"
            f"Context:\n{combined_context}\n\n"
            f"Question:\n{question}\n\n"
           
        )
        return prompt

    def generate_answer(self, prompt: str, max_length: int = 2000048) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model.generate(
              **inputs,
               
              
                temperature=0.9,             # Lower = more deterministic
                do_sample=True
               

        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)