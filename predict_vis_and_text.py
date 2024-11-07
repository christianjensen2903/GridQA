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


# PREDICTION_FILE = "predictions_gpt4_mini_vis_and_text.json"
# PREDICTION_FILE = "predictions_gemini-1.5-flash_vis_and_text.json"
PREDICTION_FILE = "predictions_gemini-1.5-pro_vis_and_text.json"
# llm = GPT4(mini=False)
# llm = Claude(model="claude-3-5-haiku-20241022")
llm = Gemini(model="gemini-1.5-pro")
formatter = Formatter()

# Set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)


def load_dataset(path: str) -> list[Sample]:
    with open(path, "r") as f:
        samples = [Sample.model_validate(sample) for sample in json.load(f)]
        return samples


dataset = load_dataset("dataset.json")

letter_mapping = {
    1: "B",
    2: "G",
    3: "R",
    4: "Y",
    5: "P",
}


def create_comparison_image(grid_before, grid_after):

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].set_title("Before")
    axs[1].set_title("After")
    # grid_shape_x, grid_shape_y = np.shape(grid_before)
    # axs[0].set_xticks(np.arange(0, grid_shape_x)+0.5)
    # axs[0].set_yticks(np.arange(0, grid_shape_y)+0.5)
    # axs[1].set_xticks(np.arange(0, grid_shape_x)+0.5)
    # axs[1].set_yticks(np.arange(0, grid_shape_y)+0.5)
    # axs[0].grid(True)
    # axs[1].grid(True)
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


predictions: dict[str, str] = {}
correct = 0
for sample in tqdm(dataset):

    prompt = f"""
Your are given a grid with one or more shapes in it.

The black shape (represented by B's) can be rotated (90, 180, 270 degrees), flipped horizontally, shifted (left, right, up, down), or left static.

Your task is to determine the transformation that was applied to the shape.

Here is the grid before the transformation:
{formatter.grid_to_text(np.array(sample.grid_before), letter_mapping)}

Here is the grid after the transformation:
{formatter.grid_to_text(np.array(sample.grid_after), letter_mapping)}

Your are also given a picture with the two grids before and after the transformation.

Your answer should be one of the following: rotate, flip, shift, static
Only answer with either rotate, flip, shift, or static. Nothing else.
"""

    # Generate the base64 encoded comparison image
    encoded_image = create_comparison_image(sample.grid_before, sample.grid_after)
    response = llm.generate(prompt, image=encoded_image, temperature=0.0, seed=seed)
    word = response.replace("\n", " ").split(" ")[0]
    predictions[sample.uuid] = word
    if response == sample.transformation.type.value:
        correct += 1

print(f"Accuracy: {correct / len(dataset)}")
# Save predictions
with open(PREDICTION_FILE, "w") as f:
    json.dump(predictions, f)
