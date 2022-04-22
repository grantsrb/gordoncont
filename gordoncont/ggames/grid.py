import numpy as np
import math
from gordoncont.ggames.constants import PLAYER, TARG, PILE, ITEM, DIVIDER, BUTTON, OBJECT_TYPES, STAY, UP, RIGHT, DOWN, LEFT, DIRECTIONS, COLORS, EVENTS, STEP, BUTTON, FULL, DEFAULT

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
    def __init__(self,
                 grid_size: int or tuple=(31,31),
                 pixel_density: int=1,
                 divide: bool=True,
                 egocentric: bool=False):
        """
        Args:
          grid_size: int or tuple (n_row, n_col)
            the dimensions of the grid. An integer arg creates a square,
            otherwise a tuple creates an n_row X n_col grid.
            Height X Width
          pixel_density: int
            the number of pixels per coordinate. If the argument is
            greater than 1, the coordinate is filled in the upper most
            left corner while leaving one pixel at the lower and
            rightmost boundaries blank.
          divide: bool
            if true, a divider is drawn horizontally across the middle 
            of the grid. If even number of rows, the divder is rounded
            up after dividing the height by two
          egocentric: bool
              determines if perspective of the game will be centered
              on the player. If true, the perspective is centered on
              the player. It is important to note that the
              observations double in size to ensure that all info in
              the game is always accessible.
        """
        self._divided = divide
        if type(grid_size) == int:
            self._grid_size = (grid_size, grid_size)
        else:
            self._grid_size = grid_size
        self._pixel_density = pixel_density
        self._grid = self.make_grid(self._divided)
        self._egocentric = egocentric
    
    @property
    def density(self):
        """
        Returns:
          pixel_density: int
            the number of pixels per unit
        """
        return self._pixel_density
    
    @property
    def egocentric(self):
        """
        Returns:
          is_egocentric: int
            if true, then get_grid will return an egocentric perspective
            centered on the argued coordinate.
        """
        return self._egocentric
    
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
    def pixel_shape(self):
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

    @property
    def middle_row(self):
        """
        This row is not included in the "playable space" when the grid
        is divided
        """
        half = self.shape[0]/2
        if self.shape[0] % 2 == 0: return half + 1
        else: return int(half)

    def units2pixels(self, coord):
        """
        Converts coordinate units to pixels
        
        Args:
          coord: int or array like (row from top, column from left)
            the coord is the coordinate on the grid in terms of grid
            units if an int is argued, only that converted value is
            returned
        """
        if type(coord) == int or type(coord) == float:
          return int(coord*self.density)
        return (
          int(coord[0]*self.density),
          int(coord[1]*self.density)
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
          int(pixel_coord[0]/self.density),
          int(pixel_coord[1]/self.density)
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
        self._grid = np.zeros(self.pixel_shape).astype(np.float)
        self._grid = self._grid + COLORS[DEFAULT]
        if do_divide:
            self.draw_divider()
        return self._grid
    
    def reset(self):
        """
        Resets the grid to the initial specifications with new
        reference to the grid.
        """
        self._grid = self.make_grid(do_divide=self.is_divided)

    def clear_unit(self, coord):
        """
        Clears a single coordinate in place. More efficient than using
        draw to draw zeros
        
        Args:
          coord: list like (row, col)
        """
        prow,pcol = self.units2pixels(coord)
        self._grid[prow:prow+self.density, pcol:pcol+self.density] = COLORS[DEFAULT]
    
    def clear_playable_space(self):
        """
        Clears the playable space of the grid in place. This means it
        zeros all information above the dividing line. If you want to 
        clear the whole grid in place, use self.clear
        """
        middle = self.units2pixels(self.middle_row)
        self._grid[:middle,:] = COLORS[DEFAULT]
    
    def clear(self, remove_divider=False):
        """
        Clears the whole grid in place.

        Args:
            remove_divider: bool
                if true, the divider is wiped from the grid as well.
                only applies if self.is_divided is true
        """
        self._grid[:,:] = COLORS[DEFAULT]
        if self.is_divided and not remove_divider:
            self.draw_divider()
    
    def draw(self, coord: tuple, color: float, add_color: bool=True):
        """
        This function handles the actual drawing on the grid. The argued
        color is drawn to the specified coordinate. Can add the argued
        color to the existing value or can replace the existing color
        with the argued depending on the value of add_color.

        Args:
          coord: array like of length 2 (row from top, column from left)
            the coord is the coordinate on the grid in terms of grid
            units
          color: float
            the value that should be drawn to the coordinate
          add_color: bool default True
            if true, the argued color is added to the existing value
            rather than replacing it.
        """
        # Coordinates that are off the grid are simply not drawn
        if not self.is_inbounds(coord): return
        row,col = self.units2pixels(coord)
        draw_space = max(1,self.density-1)
        if add_color:
            self._grid[row:row+draw_space, col:col+draw_space] += color
        else:
            self._grid[row:row+draw_space, col:col+draw_space] = color
    
    def slice_draw(self,
                   coord0: tuple,
                   coord1: tuple,
                   color: float,
                   add_color: bool=True):
        """
        Slice draws the color across a range of coordinates. It acts
        much like a numpy slice:
        
          numpy_array[row0:row1, col0:col1] = color
        
        Args:
          coord0: list like (row0, col0) (unit values)
          coord1: list like (row1, col1) (unit values) (non-inclusive)
          color: float
          add_color: bool
            if true, the argued color is added to the existing color
            instead of replacing it.
                ndarray[row0:row1, col0:col1] += color
        """
        coord0 = (max(0, coord0[0]), max(0,coord0[1]))
        coord1 = (
            min(self.shape[0], coord1[0]),
            min(self.shape[1],coord1[1])
        )
        row0,col0 = coord0
        row1,col1 = coord1
        if row0 > row1 or col0 > col1: return
        elif row0 == row1 and col0 == col1:
            self.draw(coord0, color, add_color=add_color)
            return
        elif row0 == row1:
            row1 += 1
            coord1 = (row1, col1)
        elif col0 == col1:
            col1 += 1
            coord1 = (row1, col1)
        
        # Make unit
        unit = np.zeros((self.density, self.density)) + COLORS[DEFAULT]
        unit = unit.astype(np.float)
        draw_space = max(self.density-1, 1)
        unit[0:draw_space, 0:draw_space] = color
        # Tile the unit
        n_row = int(coord1[0]-coord0[0])
        n_col = int(coord1[1]-coord0[1])
        tiles = np.tile(unit, (n_row, n_col))
        # Draw to the grid
        pxr0, pxc0 = self.units2pixels(coord0)
        pxr1, pxc1 = self.units2pixels(coord1)
        if add_color:
            self._grid[pxr0:pxr1, pxc0:pxc1] += tiles
        else:
            self._grid[pxr0:pxr1, pxc0:pxc1] = tiles

    def draw_divider(self):
        """
        Draws a divider across the middle row of the grid. Rounds up if
        even number of rows in grid.
        """
        middle = self.middle_row
        edge = self.shape[1]
        self.slice_draw(
            (middle, 0),
            (middle, edge),
            color=COLORS[DIVIDER],
            add_color=False
        )

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
        return row >= 0 and row < self.middle_row
    
    def is_inhalfbounds(self, coord):
        """
        Takes a coord and determines if it is within the divided
        boundaries of the grid.
        
        Args:
          coord: list like (row, col)
            the coordinate in grid units
        """
        row,col = coord
        return self.row_inhalfbounds(row) and self.col_inbounds(col)

    def is_playable(self, coord):
        """
        Takes a coord and determines if it is within the divided
        boundaries of the grid if the grid is divided, or simply
        within the boundaries of the grid if the grid is not divided.

        Args:
          coord: list like (row, col)
            the coordinate in grid units
        """
        if self.is_divided: return self.is_inhalfbounds(coord)
        return self.is_inbounds(coord)

    def clip_to_playable(self, coord):
        """
        Takes a coord and clips it to be within the playable boundaries
        of the grid.

        Args:
          coord: list like (row, col)
            the coordinate in grid units
        """
        if self.is_divided:
            row = min(max(0, coord[0]), self.middle_row-1)
        else:
            row = min(max(0, coord[0]), self.shape[0]-1)
        col = min(max(0,coord[1]), self.shape[1]-1)
        return (row,col)

    def is_below_divider(self, coord):
        """
        Takes a coord and determines if it is within the divided
        boundaries of the grid if the grid is divided, or simply
        within the boundaries of the grid if the grid is not divided.

        Args:
          coord: list like (row, col)
            the coordinate in grid units
        """
        row, col = coord
        if self.is_divided:
            row_inbounds = row > self.middle_row and row < self.shape[0]
            return row_inbounds and self.col_inbounds(col)
        return self.is_inbounds(coord)

    def get_grid(self, coord=None):
        """
        Returns a perspective of the grid dependent on the egocentric
        flag

        Args:
            coord: None or list like (row, col)
                if egocentric, this is the center coordinate for the
                returned perspective
        """
        if self.egocentric:
            return self.get_egocentric_perspective(coord)
        return self.grid

    def get_egocentric_perspective(self, coord):
        """
        Returns a perspective of the game that is centered
        on the argued coord. The returned observation is also double
        the original size of the grid to ensure that all info in
        the game is always accessible.
        """
        shape = self.pixel_shape
        pad = np.zeros((shape[0]*2, shape[1]*2))
        r_low =  shape[0]//2
        r_high = shape[0]-r_low
        c_low =  shape[1]//2
        c_high = shape[1]-c_low

        if coord is None: p = (r_low, c_low)
        else: p = self.units2pixels(coord)
        rdev = p[0]-r_low
        cdev = p[1]-c_low
        pad[r_low-rdev:-r_high-rdev,c_low-cdev:-c_high-cdev]=self.grid
        return pad


