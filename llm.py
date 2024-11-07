from openai import OpenAI
from abc import ABC, abstractmethod
import dotenv
from anthropic import Anthropic
import google.generativeai as genai
import os
from io import BytesIO
from PIL import Image
import base64


class LLM(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        image: str | None = None,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> str:
        pass


class GPT4(LLM):
    def __init__(self, model: str = "gpt-4o"):
        dotenv.load_dotenv()
        self.client = OpenAI()
        self.model = model

    def generate(
        self,
        prompt: str,
        image: str | None = None,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        if image:
            messages[0]["content"] = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image}"},
                },
            ]

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            seed=seed,
            messages=messages,
        )

        assert response.choices[0].message.content
        return response.choices[0].message.content


class Claude(LLM):
    def __init__(self, model="claude-3-5-sonnet-latest"):
        dotenv.load_dotenv()
        self.client = Anthropic()
        self.model = model

    def generate(
        self,
        prompt: str,
        image: str | None = None,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        if image:
            messages[0]["content"] = [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image,
                    },
                },
            ]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=10,
            temperature=temperature,
            messages=messages,
        )
        return response.content[-1].text


class Gemini(LLM):
    def __init__(self, model="gemini-1.5-flash"):
        dotenv.load_dotenv()
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model

    def generate(
        self,
        prompt: str,
        image: str | None = None,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> str:
        model = genai.GenerativeModel(self.model)
        content = [prompt]

        if image:
            image_data = base64.b64decode(image)
            data = Image.open(BytesIO(image_data))
            content = [data, "\n\n", prompt]

        result = model.generate_content(
            content,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
            ),
        )

        return result.text


if __name__ == "__main__":
    generator = Gemini()
    prompt = "What color is the top-left cell?"
    response = generator.generate(prompt)
    print(response)
