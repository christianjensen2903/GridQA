import llm 
import torch
from formatter import Formatter
import random
import numpy as np
from generate_dataset import Sample
import json
from tqdm import tqdm
from transformers import pipeline



pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-3B")
# Set up pipeline with CPU-only quantization
# pipe = pipeline(
#     "text-generation",
#     model="meta-llama/Llama-3.1-8B",
#     device="cpu",  # Explicitly set to CPU
#     # torch_dtype=torch.float16,
#     torch_dtype=torch.bfloat16
# )


formatter = Formatter()
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

predictions: dict[str, str] = {}
correct = 0
for sample in tqdm(dataset):

    prompt = f"""
Your are given a grid with one or more shapes in it.

The black shape (represented by S's) can be rotated (90, 180, 270 degrees), flipped horizontally, shifted (left, right, up, down), or left static.

Your task is to determine the transformation that was applied to the shape.

Here is the grid before the transformation:
{formatter.grid_to_text(np.array(sample.grid_before), letter_mapping)}

Here is the grid after the transformation:
{formatter.grid_to_text(np.array(sample.grid_after), letter_mapping)}

Your answer should be one of the following transformation: rotate, flip, shift or static
Only answer with one of the words rotate, flip, shift, or static. Nothing else.
The one correct transformation is: 
"""
    response = pipe(prompt, max_new_tokens = 4)[0]['generated_text'][len(prompt):]
    word = response.replace("\n", " ").split(" ")[0]
    print("\nlabel:", sample.transformation.type.value)
    print("prediction:", word)
    print("response:" , response)
    # print(formatter.grid_to_text(np.array(sample.grid_before), letter_mapping) + "\n" + formatter.grid_to_text(np.array(sample.grid_after), letter_mapping))
    predictions[sample.uuid] = word
    if response == sample.transformation.type.value:
        correct += 1

print(f"Accuracy: {correct / len(dataset)}")
with open("predictions_Llama3B.json", "w") as f:
    json.dump(predictions, f)