from llm import LLM, GPT4
from formatter import Formatter
from generate_shape import generate_rotational_shape
import random
import numpy as np
import math


llm = GPT4(mini=False)
formatter = Formatter()

# Set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)

n = 10
grid_size = 15
shape_size = 5

correct = 0
for i in range(n):

    shape_before = generate_rotational_shape(shape_size)

    options = ["rotate", "flip", "shift", "static"]

    choice = random.choice(options)

    shape_after = shape_before.copy()
    if choice == "rotate":
        degrees = random.choice([90, 180, 270])
        shape_after = np.rot90(shape_after, degrees)
    elif choice == "flip":
        shape_after = np.fliplr(shape_after)

    shape_h_before, shape_w_before = shape_before.shape
    shape_h_after, shape_w_after = shape_after.shape

    center = grid_size // 2
    x_offset = 0
    y_offset = 0

    if choice == "shift":
        direction = random.choice(["left", "right", "up", "down"])
        offset = random.randint(2, 5)
        if direction == "left":
            x_offset = -offset
        elif direction == "right":
            x_offset = offset
        elif direction == "up":
            y_offset = -offset
        elif direction == "down":
            y_offset = offset

    grid_before = np.zeros((grid_size, grid_size))
    grid_before[
        center - shape_h_before // 2 : center + math.ceil(shape_h_before / 2),
        center - shape_w_before // 2 : center + math.ceil(shape_w_before / 2),
    ] = shape_before
    grid_after = np.zeros((grid_size, grid_size))

    grid_after[
        center
        - shape_h_after // 2
        + y_offset : center
        + math.ceil(shape_h_after / 2)
        + y_offset,
        center
        - shape_w_after // 2
        + x_offset : center
        + math.ceil(shape_w_after / 2)
        + x_offset,
    ] = shape_after

    prompt = f"""
Your are given a grid with a shape in it.

The shape can be rotated (90, 180, 270 degrees), flipped horizontally, shifted (left, right, up, down), or left static.

The shape is represented by #'s.

Your task is to determine the transformation that was applied to the shape.

Here is the grid before the transformation:
{formatter.grid_to_text(grid_before)}

Here is the grid after the transformation:
{formatter.grid_to_text(grid_after)}

Your answer should be one of the following: rotate, flip, shift, static
Only answer with either rotate, flip, shift, or static. Nothing else.
"""

    response = llm.generate(prompt, temperature=0.0, seed=seed)
    print(choice, response)
    if response == choice:
        correct += 1

print(f"Accuracy: {correct / n}")
