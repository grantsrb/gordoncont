import numpy as np
import math

"""
The grid class handles the drawing of objects to the image. It enables
setting the pixel density, size of the grid, preventing out of bounds 
erros, and protecting the image info.

Coordinates are recorded as (Row, Col) to keep in line with ndarray
thinking. For instance, the coordinate (1,2) will get drawn one row
down from the top and 3 rows over from the left. Coordinates refer
to grid units, NOT PIXELS.
"""
class Grid:
  DEFAULT_COLOR = 0

  def __init__(self,
               grid_size: int or tuple=(31,31),
               pixel_density: int=1,
               divide: bool=True):
    """
    Args:
      grid_size: int or tuple (n_row, n_col)
        the dimensions of the grid. An integer arg creates a square,
        otherwise a tuple creates an n_row X n_col grid. HeightXWidth
      pixel_density: int
        the number of pixels per coordinate. If the argument is greater
        than 1, the coordinate is filled in the upper most left corner
        while leaving one pixel at the lower and rightmost boundaries
        blank.
      divide: bool
        if true, a divider is drawn horizontally across the middle of
        the grid. If even number of rows, the divder is rounded up
        after dividing the height by two
    """
    self._divided = divide
    if type(grid_size) == int:
      self._grid_size = (grid_size, grid_size)
    else:
      self._grid_size = grid_size
    self._pixel_density = pixel_density
    self._grid = self.make_grid(self._divided)

  @property
  def density(self):
    """
    Returns:
      pixel_density: int
        the number of pixels per unit
    """
    return self._pixel_density

  @property
  def is_divided(self):
    """
    Returns:
      bool
        true if grid is divided
    """
    return self._divided

  @property
  def shape(self):
    """
    Returns the shape of the grid in terms of units

    Returns:
      _grid_size: tuple (n_row, n_col)
    """
    return self._grid_size

  @property
  def pixel_shape():
    """
    Returns the shape of the grid in terms of pixels rather than grid
    units

    Returns:
      pixel_shape: tuple (n_row, n_col)
    """
    return self.units2pixels(self.shape)

  @property
  def grid(self):
    return self._grid.copy()

  def units2pixels(self, coord):
    """
    Converts coordinate units to pixels

    Args:
      coord: int or array like (row from top, column from left)
        the coord is the coordinate on the grid in terms of grid units
        if an int is argued, only that converted value is returned
    """
    if type(coord) == int:
      return coord*self.density,
    return (
      coord[0]*self.density,
      coord[1]*self.density
    )

  def pixels2units(self, pixel_coord):
    """
    Converts a pixel coordinate to the unit coordinate. Rounds down
    to nearest coord

    Args:
      pixel_coord: array like (row from top, column from left)
        this is the coordinate on the grid in terms of pixels
    """
    shape = self.shape
    return (
      int(pixel_coord[0]/shape[0]),
      int(pixel_coord[1]/shape[1]
    )

  def make_grid(self, do_divide=True):
    """
    Creates the grid to the specified unit dimensions, each unit
    containing a square of pixels with height and width equal to the
    pixel density.

    Args:
      do_divide: bool
        if true, a divider is drawn across the middle of the grid.
    Returns:
      grid: ndarry (H,W)
        a numpy array representing the grid
    """
    self._grid = np.zeros(self.pixel_shape) + DEFAULT_COLOR
    if do_divide:
      middle = math.ceil(self.shape[0]/2)
      edge = self.shape[1]
      self.slice_draw((middle, 0), (middle, edge))

  def clear_unit(self, coord):
    """
    Clears a single coordinate in place. More efficient than using
    draw to draw zeros

    Args:
      coord: list like (row, col)
    """
    prow,pcol = self.units2pixels(coord)
    self._grid[prow:prow+self.density, pcol:pcol+self.density] = 0

  def clear_divided(self):
    """
    Clears the playable space of the grid in place. This means it
    zeros all information above the dividing line. If you want to 
    clear the whole grid in place, use self.clear
    """
    middle = math.ceil(self.shape[0]/2)
    self._grid[0:middle,:] = 0

  def clear(self):
    """
    Clears the whole grid in place.
    """
    self._grid[:,:] = 0

  def draw(self, coord: list like, color: float):
    """
    This function handles the actual drawing on the grid. The argued
    color is drawn to the specified coordinate.

    Args:
      coord: array like of length 2 (row from top, column from left)
        the coord is the coordinate on the grid in terms of grid units
      color: float
        the value that should be drawn to the coordinate
    """
    assert len(coord) == 2
    # Coordinates that are off the grid are simply not drawn
    if not (coord[0] < 0 or coord[0] >= self.shape[0]): return
    elif not (coord[1] < 0 or coord[1] >= self.shape[1]): return
    row,col = self.units2pixels(coord)
    density = max(1,self.density-1)
    self._grid[row:row+density, col:col+density] = color

  def slice_draw(self,
                 coord0: list like,
                 coord1: list like,
                 color: float):
    """
    Slice draws the color across a range of coordinates. It acts much
    like a numpy slice:

      numpy_array[row0:row1, col0:col1] = color

    Does not do anything if either row0 or col0 are greater than
    row1 or col1 respectively.

    Args:
      coord0: list like (row0, col0) (unit values)
      coord1: list like (row1, col1) (unit values) (non-inclusive)
      color: float
    """
    row0,col0 = coord0
    row1,col1 = coord1
    if row0 > row1 or col0 > col1: return
    elif row0 == row1 and col0 == col1:
      self.draw(coord0, color)
      return
    elif row0 == row1:
      row1 += 1
      coord1 = (row1, col1)
    elif col0 == col1:
      col1 += 1
      coord1 = (row1, col1)

    # Make unit
    unit = np.zeros((self.density, self.density)) + DEFAULT_COLOR
    unit = unit.astype(np.float)
    unit[0:additive, 0:additive] = color
    # Tile the unit
    n_row = coord1[0]-coord0[0]
    n_col = coord1[1]-coord0[1]
    tiles = np.tile(unit, (n_row, n_col))
    # Draw to the grid
    pxr0, pxc0 = self.units2pixels(coord0)
    pxr1, pxc1 = self.units2pixels(coord1)
    self._grid[pxr0:pxr1, pxc0:pxc1] = tiles

  def row_inbounds(self, row):
    """
    Determines if the argued row is within the bounds of the grid

    Args:
      row: int
    Returns:
      inbounds: bool
        true if row is in bounds
    """
    return row >= 0 and row < self.shape[0]

  def col_inbounds(self, col):
    """
    Determines if the argued col is within the bounds of the grid

    Args:
      col: int
    Returns:
      inbounds: bool
        true if col is in bounds
    """
    return col >= 0 and col < self.shape[1]

  def is_inbounds(self, coord):
    """
    Takes a coord and determines if it is within the boundaries of
    the grid.

    Args:
      coord: list like (row, col)
        the coordinate in grid units
    """
    row, col = coord
    return self.row_inbounds(row) and self.col_inbounds(col)

  def row_inhalfbounds(self, row):
    """
    Determines if the row is within the divided boundaries of
    the grid.

    Args:
      row: int
    Returns:
      inbounds: bool
        true if the argued row is visually above the divided bounds
        of the grid
    """
    return row >= 0 and row < math.ceil(self.shape[0]/2)

