from llm import GPT4, LLM, Claude, Gemini
from formatter import Formatter
import random
import numpy as np
from generate_dataset import Sample
import json
from tqdm import tqdm
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import prompts
from typing import Literal


def load_dataset(path: str) -> list[Sample]:
    with open(path, "r") as f:
        samples = [Sample.model_validate(sample) for sample in json.load(f)]
        return samples


def create_comparison_image(grid_before, grid_after):

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].set_title("Before")
    axs[1].set_title("After")
    axs[0].imshow(grid_before, cmap="Greys", interpolation="none")
    axs[1].imshow(grid_after, cmap="Greys", interpolation="none")
    axs[0].tick_params(labelbottom=False, labelleft=False)
    axs[1].tick_params(labelbottom=False, labelleft=False)

    # Add an arrow between the two grids to show transition
    fig.text(0.5, 0.5, "→", ha="center", va="center", fontsize=30)

    # Save the figure to a BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="PNG", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image_base64


dataset = load_dataset("dataset.json")

letter_mapping = {
    1: "R",
    2: "G",
    3: "B",
    4: "Y",
    5: "P",
}

formatter = Formatter()


def grid_to_text(grid: list[list[int]]) -> str:
    return formatter.grid_to_text(np.array(grid), letter_mapping)


def predict(
    model: LLM, input_type: Literal["image", "text", "image-text"], seed: int = 42
) -> dict[str, str]:

    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    predictions: dict[str, str] = {}
    for sample in tqdm(dataset):

        if input_type == "image-text":
            prompt = prompts.image_text_prompt
        elif input_type == "image":
            prompt = prompts.image_prompt
        elif input_type == "text":
            prompt = prompts.text_prompt

        prompt = prompt.format(
            grid_before=grid_to_text(sample.grid_before),
            grid_after=grid_to_text(sample.grid_after),
        )

        image = (
            create_comparison_image(sample.grid_before, sample.grid_after)
            if input_type != "text"
            else None
        )

        response = model.generate(prompt, image=image, temperature=0.0, seed=seed)
        prediction = response.replace("\n", " ").split(" ")[0]
        predictions[sample.uuid] = prediction

    return predictions


if __name__ == "__main__":
    models = {
        "gpt4o": GPT4(model="gpt-4o-2024-08-06"),
        "gpt4-mini": GPT4(model="gpt-4o-mini-2024-07-18"),
        "claude-3.5-sonnet": Claude(model="claude-3-5-sonnet-2024102"),
        "gemini-1.5-flash": Gemini(model="gemini-1.5-flash-002"),
        "gemini-1.5-pro": Gemini(model="gemini-1.5-pro-002"),
    }

    for input_type in ["image", "text", "image-text"]:
        for model_name, model in models.items():
            predictions = predict(model, input_type)
            print(f"{model_name} {input_type}: {predictions}")

        with open(f"predictions2/{model_name}_{input_type}.json", "w") as f:
            json.dump(predictions, f)
