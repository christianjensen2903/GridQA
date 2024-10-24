import numpy as np


class Formatter:
    """
    Convert the input and output grids to ASCII format.
    """

    def char_to_text(self, char: int) -> str:
        """Convert integer to letter in the alphabet"""
        return "#" if char > 0 else " "

    def grid_to_text(self, grid: np.ndarray, letter_mapping: dict[int, str]) -> str:
        height, width = grid.shape
        grid_str = ""

        for i, row in enumerate(grid):
            grid_str += "|"
            for value in row:
                char = self.char_to_text(value)
                grid_str += (
                    f"{letter_mapping[value] if value in letter_mapping else char}|"
                )
            grid_str += "\n"

        return grid_str


if __name__ == "__main__":
    formatter = Formatter()

    # Make random 40x40 grid
    grid = np.random.randint(0, 10, (40, 40))
    print(formatter.grid_to_text(grid, {1: "R", 2: "G", 3: "B", 4: "Y", 5: "P"}))
