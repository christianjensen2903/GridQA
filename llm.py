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
        self, prompt: str, iamge=None, temperature: float = 0.0, seed: int = 42
    ) -> str:
        pass


class GPT4(LLM):
    def __init__(self, mini: bool = False):
        dotenv.load_dotenv()
        self.client = OpenAI()
        self.mini = mini

    def generate(
        self, prompt: str, image=None, temperature: float = 0.0, seed: int = 42
    ) -> str:

        if image is not None:
            response = self.client.chat.completions.create(
                model="gpt-4o" if not self.mini else "gpt-4o-mini",
                temperature=temperature,
                seed=seed,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image}"},
                            },
                        ],
                    }
                ],
            )
        else:
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

    def generate(
        self, prompt: str, image=None, temperature: float = 0.0, seed: int = 42
    ) -> str:
        if image is not None:
            response = self.client.messages.create(
                model=self.model,
                temperature=temperature,
                max_tokens=10,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image,
                                },
                            },
                        ],
                    }
                ],
            )
        else:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        return response.content[-1].text


class Gemini(LLM):
    def __init__(self, model="gemini-1.5-flash"):
        dotenv.load_dotenv()
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model

    def generate(
        self, prompt: str, image=None, temperature: float = 0.0, seed: int = 42
    ) -> str:

        model = genai.GenerativeModel(self.model)

        if image is not None:
            # data = BytesIO(image.encode())
            # myfile = genai.upload_file(
            #     data.getvalue(), mime_type="image/png", display_name="image.png"
            # )
            image_data = base64.b64decode(image)
            data = Image.open(BytesIO(image_data))
            result = model.generate_content(
                [data, "\n\n", prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                ),
            )
        else:
            result = model.generate_content(
                [prompt],
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
