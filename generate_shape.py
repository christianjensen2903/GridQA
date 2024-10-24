import random
import matplotlib.pyplot as plt
import numpy as np


def generate_random_shape(n, value: int = 1):
    grid_size = n * 2
    grid = np.zeros((grid_size, grid_size), dtype=int)

    # Start the shape from the center of the grid
    x, y = grid_size // 2, grid_size // 2
    grid[x, y] = value
    shape_coordinates = [(x, y)]

    # Directions: Up, Down, Left, Right
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while len(shape_coordinates) < n:
        # Pick a random square already in the shape
        random_square = random.choice(shape_coordinates)
        x, y = random_square

        # Pick a random direction
        dx, dy = random.choice(directions)
        new_x, new_y = x + dx, y + dy

        # Ensure the new square is within grid bounds and not already part of the shape
        if (
            0 <= new_x < grid_size
            and 0 <= new_y < grid_size
            and grid[new_x, new_y] == 0
        ):
            grid[new_x, new_y] = value
            shape_coordinates.append((new_x, new_y))

    # Reduce the grid to the size of the shape
    min_row, min_col = np.min(shape_coordinates, axis=0)
    max_row, max_col = np.max(shape_coordinates, axis=0)

    grid = grid[min_row : max_row + 1, min_col : max_col + 1]

    return grid


def transform_shape(grid):
    """
    Returns all distinct transformations (rotations and reflections) of the grid.
    Includes 4 rotations (0, 90, 180, 270 degrees) and horizontal reflection.
    """
    transformations = []
    for flip_h in [False, True]:  # Horizontal flip
        for rotation in range(4):  # 4 possible rotations
            transformed = np.rot90(grid, rotation)
            if flip_h:
                transformed = np.fliplr(transformed)

            transformations.append((flip_h, rotation, transformed))
    return transformations


def is_distinguishable(grid):
    """
    Checks if the shape is distinguishable by comparing all possible transformations.
    If any two transformations are identical, the shape is not distinguishable.
    """
    transformations = transform_shape(grid)
    # Check if all transformations are unique
    for i, t1 in enumerate(transformations):
        for t2 in transformations[i + 1 :]:
            if np.array_equal(t1[2], t2[2]):
                return False
    return True


def generate_distinguishable_shape(n, max_attempts=1000, value: int = 1):
    """
    Generates a random shape that is distinguishable by rotation and reflection.
    """
    for i in range(max_attempts):
        grid = generate_random_shape(n, value)
        if is_distinguishable(grid):
            return grid
    raise Exception(
        "Failed to generate a distinguishable shape after {} attempts".format(
            max_attempts
        )
    )


def generate_rotational_shape(n, value: int = 1):
    """
    Generate a shape that is distinguishable by rotation and have a clear center (odd width and height)
    """
    grid = generate_distinguishable_shape(n, value=value)
    if grid.shape[0] % 2 == 0 or grid.shape[1] % 2 == 0:
        return generate_rotational_shape(n, value=value)
    return grid


def plot_shape(grid):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="Greys", interpolation="none")
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == "__main__":
    random_shape_grid = generate_rotational_shape(30)

    # Show all transformations in one plot
    transformations = transform_shape(random_shape_grid)
    fig, axs = plt.subplots(2, len(transformations) // 2, figsize=(20, 10))
    print(len(transformations), len(transformations) // 2)
    for i, t in enumerate(transformations):
        print(i, i // 2, i % 2)
        axs[i % 2, i // 2].imshow(t[2], cmap="Greys", interpolation="none")
        axs[i % 2, i // 2].set_title(f"Flip: {t[0]}, Rot: {t[1]}")

    plt.tight_layout()
    plt.show()
