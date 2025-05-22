from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class Generator:
    def __init__(self, model_name="google/flan-t5-large", device=None):
        self.device = device or "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def build_prompt(self, context, question):
        combined = "\n".join(context)
        return (
            "You are an expert assistant. Provide a comprehensive, richly detailed answer "
            "of at least 150 words, including examples and explanations based strictly on the context.\n\n"
            f"Context:\n{combined}\n\n"
            f"Question:\n{question}\n\n"
            "Answer in detail:"
        )

    def generate_answer(self, prompt, **generate_kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        defaults = {
            "max_length": 512,
            "min_length": 150,
            "num_beams": 4,
            "length_penalty": 1.0,
            "early_stopping": True,
        }
        kwargs = {**defaults, **generate_kwargs}
        outputs = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

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
            "You are an expert assistant that only answers questions about quries if the relevant information is available in the given context.\n"
            "Answer strictly based on the context provided..\n"
            f"Context:\n{combined_context}\n\n"
            f"Question:\n{question}\n\n"
           
        )
        return prompt

    def generate_answer(self, prompt: str, max_length: int = 1028) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model.generate(
              **inputs,
               max_length=max_length,
              
                temperature=0.6,             # Lower = more deterministic
                do_sample=True
               

        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)