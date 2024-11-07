from generate_shape import generate_rotational_shape, generate_random_shape, plot_shape
import random
import numpy as np
import math
from pydantic import BaseModel, Field
from enum import Enum
import json
import uuid

# Set random seed
seed = 5141234  # Seed to ensure it could generate without errors
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
    uuid: str = Field(default_factory=lambda: str(uuid.uuid4()))
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


def can_place_shape(grid: np.ndarray, shape: np.ndarray, row: int, col: int) -> bool:
    shape_h, shape_w = shape.shape
    grid_h, grid_w = grid.shape

    if row + shape_h > grid_h or col + shape_w > grid_w:
        return False

    for i in range(shape_h):
        for j in range(shape_w):
            if grid[row + i, col + j] != 0 and shape[i, j] != 0:
                return False
    return True


# Function to randomly place a shape in the grid
def place_shape_randomly(grid: np.ndarray, shape: np.ndarray) -> None:
    grid_h, grid_w = grid.shape
    shape_h, shape_w = shape.shape

    placed = False
    while not placed:
        row = random.randint(0, grid_h - shape_h)
        col = random.randint(0, grid_w - shape_w)
        if can_place_shape(grid, shape, row, col):
            grid[row : row + shape_h, col : col + shape_w] = shape
            placed = True


def generate_dataset(
    configurations: list[Configuration], samples_per_configuration: int = 10
) -> list[Sample]:
    dataset: list[Sample] = []

    for configuration in configurations:
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
            # choice = options[i%len(options)]
            print("")
            print(random.choice(options))
            print(options[i%len(options)])

            transformation_params: dict = {}

            shape_after = shape_before.copy()
            if choice == TransformationType.ROTATE:
                n_90_flips = random.choice([1, 2, 3])
                shape_after = np.rot90(shape_after, n_90_flips)
                transformation_params["degrees"] = int(360 - n_90_flips * 90)
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

            grid = np.zeros((grid_size, grid_size))

            # Add irrelevant shapes to the grid at random locations
            if number_of_shapes > 1:
                for i in range(2, number_of_shapes + 1):
                    extra_shape = generate_random_shape(shape_size, value=i)
                    place_shape_randomly(grid, extra_shape)

            # Create empty grids
            grid_before = grid.copy()
            grid_after = grid.copy()

            # Place the main shape at the center
            grid_before[
                center - shape_h_before // 2 : center + math.ceil(shape_h_before / 2),
                center - shape_w_before // 2 : center + math.ceil(shape_w_before / 2),
            ] = shape_before

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

            # Add sample to the dataset
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

    return dataset


if __name__ == "__main__":

    dataset = generate_dataset(configurations, samples_per_configuration=20)
    print(f"Generated {len(dataset)} samples")

    with open("dataset.json", "w") as f:
        json.dump([sample.model_dump(mode="json") for sample in dataset], f)
