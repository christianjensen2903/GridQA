from openai import OpenAI
from abc import ABC, abstractmethod
import dotenv
from anthropic import Anthropic


class LLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass


class GPT4(LLM):
    def __init__(self, mini: bool = False):
        dotenv.load_dotenv()
        self.client = OpenAI()
        self.mini = mini

    def generate(self, prompt: str, temperature: float = 0.0, seed: int = 42) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o" if not self.mini else "gpt-4o-mini",
            temperature=temperature,
            seed=seed,
            messages=[{"role": "user", "content": prompt}],
        )
        assert response.choices[0].message.content is not None
        return response.choices[0].message.content


class Claude(LLM):
    def __init__(self, model="claude-3-5-sonnet-latest"):
        dotenv.load_dotenv()
        self.client = Anthropic()
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.0, seed: int = 42) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[-1].text


if __name__ == "__main__":
    generator = Claude()
    prompt = "What color is the top-left cell?"
    response = generator.generate(prompt)
    print(response)
