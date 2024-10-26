from io import BytesIO
import base64
import numpy as np
from PIL import Image, ImageDraw
import cv2

rgb_lookup = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (128, 0, 128),
}


def calculate_grid_size(
    grid: np.ndarray, cell_size: int, edge_size: int
) -> tuple[int, int]:
    height, width = grid.shape
    new_height = height * (cell_size + edge_size) + edge_size
    new_width = width * (cell_size + edge_size) + edge_size
    return new_height, new_width


def grid_to_rgb(grid: np.ndarray, cell_size: int, edge_size: int):
    grid_height, grid_width = calculate_grid_size(grid, cell_size, edge_size)

    edge_color = (85, 85, 85)  # Grey edge color

    rgb_grid = np.full((grid_height, grid_width, 3), edge_color, dtype=np.uint8)

    # Fill in the cells with the appropriate colors
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            color = rgb_lookup[int(grid[i, j])]
            y = i * (cell_size + edge_size) + edge_size
            x = j * (cell_size + edge_size) + edge_size
            rgb_grid[y : y + cell_size, x : x + cell_size] = color

    return rgb_grid


def add_grid_border(rgb_grid: np.ndarray, border_size: int):
    grid_height, grid_width = rgb_grid.shape[:2]

    total_height = grid_height + border_size * 2
    total_width = grid_width + border_size * 2
    rgb_grid_border = np.full(
        (total_height, total_width, 3), (255, 255, 255), dtype=np.uint8
    )

    # Center the grid
    rgb_grid_border[border_size:-border_size, border_size:-border_size] = rgb_grid

    return rgb_grid_border


def create_rgb_grid(grid: np.ndarray, cell_size: int, edge_size: int):
    rgb_grid = grid_to_rgb(grid, cell_size, edge_size)
    rgb_grid = add_grid_border(rgb_grid, border_size=cell_size)

    return rgb_grid


def draw_arrow(arrow_size: int = 20):

    arrow_head_size = arrow_size
    image_size = (arrow_head_size * 2, arrow_head_size)

    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)

    # Define the arrow parameters
    start_point = (0, arrow_head_size // 2)  # Start of the arrow
    end_point = (arrow_head_size, arrow_head_size // 2)  # End of the arrow (head)

    # Draw the arrow line
    draw.line([start_point, end_point], fill="black", width=5)

    # Draw the arrowhead (triangle)
    draw.polygon(
        [(arrow_size, 0), (arrow_size, arrow_size), (arrow_size * 2, arrow_size // 2)],
        fill="black",
    )

    return image


def add_arrow_between_images(
    image1: np.ndarray, image2: np.ndarray, arrow_size: int = 20
):
    # Convert the images to PIL for easier manipulation
    image1_pil = Image.fromarray(image1)
    image2_pil = Image.fromarray(image2)

    image_height = max(image1_pil.height, image2_pil.height)
    center_height = int(image_height // 2)

    arrow_image = draw_arrow(arrow_size=arrow_size)
    arrow_height = arrow_image.height

    # Combine the images with the arrow in between

    side_padding = 10
    total_width = (
        image1_pil.width + arrow_image.width + image2_pil.width + side_padding * 2
    )
    new_image = Image.new("RGB", (total_width, image_height), (255, 255, 255))
    new_image.paste(image1_pil, (0, center_height - image1_pil.height // 2))
    new_image.paste(
        arrow_image,
        (image1_pil.width + side_padding, center_height - arrow_height // 2),
    )
    new_image.paste(
        image2_pil,
        (
            image1_pil.width + arrow_image.width + side_padding * 2,
            center_height - image2_pil.height // 2,
        ),
    )

    return np.array(new_image)


def show_input_output_side_by_side(
    input_grid: np.ndarray, output_grid: np.ndarray, cell_size: int, edge_size: int
):
    input_rgb = create_rgb_grid(input_grid, cell_size, edge_size)
    output_rgb = create_rgb_grid(output_grid, cell_size, edge_size)
    combined_image = add_arrow_between_images(input_rgb, output_rgb)

    return Image.fromarray(combined_image)


def grid_to_base64_png_oai_content(grid: np.ndarray, cell_size: int, edge_size: int):

    rgb_grid = create_rgb_grid(grid, cell_size, edge_size)
    image = Image.fromarray(rgb_grid, "RGB")

    output = BytesIO()
    image.save(output, format="PNG")
    base64_png = base64.b64encode(output.getvalue()).decode("utf-8")

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_png}",
        },
    }


def demonstrations_to_oai_content(input_grid: np.ndarray, output_grid: np.ndarray):
    image = show_input_output_side_by_side(
        input_grid, output_grid, cell_size=30, edge_size=3
    )
    output = BytesIO()
    image.save(output, format="PNG")
    base64_png = base64.b64encode(output.getvalue()).decode("utf-8")

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_png}",
        },
    }


if __name__ == "__main__":

    initial_values = np.random.randint(0, 6, (15, 15))
    output_values = np.random.randint(0, 6, (15, 15))

    image = show_input_output_side_by_side(
        initial_values, output_values, cell_size=30, edge_size=3
    )
    image.show()
