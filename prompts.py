image_prompt = """
Your are given a picture with two grids that contain a shape. The shape in the right hand side grid is equal to the shape in the left hand side 
grid - except for the fact that it has been either rotated (90, 180, 270 degrees), flipped horizontally, shifted (left, right, up, down), or left static.

Your task is to determine the type transformation that was applied to the shape on the left in order to get the shape on the right.

Your answer should be one of the following: rotate, flip, shift, static
Only answer with either rotate, flip, shift, or static. Nothing else.
"""


text_prompt = """
Your are given a grid with one or more shapes in it.

The red shape (represented by R's) can be rotated (90, 180, 270 degrees), flipped horizontally, shifted (left, right, up, down), or left static.

Your task is to determine the transformation that was applied to the shape.

Here is the grid before the transformation:
{grid_before}

Here is the grid after the transformation:
{grid_after}

Your answer should be one of the following: rotate, flip, shift, static
Only answer with either rotate, flip, shift, or static. Nothing else.
"""


image_text_prompt = """
Your are given a grid with one or more shapes in it.

The black shape (represented by B's) can be rotated (90, 180, 270 degrees), flipped horizontally, shifted (left, right, up, down), or left static.

Your task is to determine the transformation that was applied to the shape.

Here is the grid before the transformation:
{grid_before}

Here is the grid after the transformation:
{grid_after}

Your are also given a picture with the two grids before and after the transformation.

Your answer should be one of the following: rotate, flip, shift, static
Only answer with either rotate, flip, shift, or static. Nothing else.
"""
