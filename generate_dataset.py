from generate_shape import generate_rotational_shape


from llm import LLM, GPT4
from formatter import Formatter
from generate_shape import generate_rotational_shape
import random
import numpy as np
import math
from pydantic import BaseModel
from enum import Enum
import json

# Set random seed
seed = 2  # Seed to ensure it could generate without errors
random.seed(seed)
np.random.seed(seed)


class Configuration(BaseModel):
    shape_size: int
    grid_size: int
    number_of_shapes: int


class TransformationType(Enum):
    ROTATE = "rotate"
    FLIP = "flip"
    SHIFT = "shift"
    STATIC = "static"


class Transformation(BaseModel):
    type: TransformationType
    transformation_params: dict


class Sample(BaseModel):
    configuration: Configuration
    grid_before: list[list[int]]
    grid_after: list[list[int]]
    transformation: Transformation


configurations = (
    [
        Configuration(shape_size=size, grid_size=20, number_of_shapes=1)
        for size in range(10, 31, 5)  # the 5 shape size is generated below
    ]
    + [
        Configuration(shape_size=5, grid_size=size, number_of_shapes=1)
        for size in range(10, 31, 5)
    ]
    + [
        Configuration(shape_size=5, grid_size=20, number_of_shapes=n)
        for n in range(2, 6)  # The 1 shape is generated above
    ]
)

dataset: list[Sample] = []


samples_per_configuration = 10


for configuration in configurations:
    print(f"Generating samples for configuration:")
    print(configuration.model_dump_json(indent=2))
    shape_size = configuration.shape_size
    grid_size = configuration.grid_size
    number_of_shapes = configuration.number_of_shapes

    for i in range(samples_per_configuration):

        shape_before = generate_rotational_shape(shape_size)

        options = [
            TransformationType.ROTATE,
            TransformationType.FLIP,
            TransformationType.SHIFT,
            TransformationType.STATIC,
        ]

        choice = random.choice(options)
        transformation_params = {}

        shape_after = shape_before.copy()
        if choice == TransformationType.ROTATE:
            degrees = random.choice([90, 180, 270])
            shape_after = np.rot90(shape_after, degrees)
            transformation_params["degrees"] = degrees
        elif choice == TransformationType.FLIP:
            shape_after = np.fliplr(shape_after)

        shape_h_before, shape_w_before = shape_before.shape
        shape_h_after, shape_w_after = shape_after.shape

        center = grid_size // 2
        x_offset = 0
        y_offset = 0

        if choice == TransformationType.SHIFT:
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

            transformation_params["direction"] = direction
            transformation_params["offset"] = offset

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

        dataset.append(
            Sample(
                configuration=configuration,
                grid_before=grid_before.tolist(),
                grid_after=grid_after.tolist(),
                transformation=Transformation(
                    type=choice,
                    transformation_params=transformation_params,
                ),
            )
        )

print(f"Generated {len(dataset)} samples")

with open("dataset.json", "w") as f:
    json.dump([sample.model_dump(mode="json") for sample in dataset], f)
