import os, subprocess, time, signal
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gordongames.envs.ggames import Discrete
from gordongames.envs.ggames.controllers import *
from gordongames.envs.ggames.constants import STAY, ITEM, TARG, PLAYER, PILE, BUTTON
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: see matplotlib documentation for installation https://matplotlib.org/faq/installing_faq.html#installation".format(e))

class GordonGame(gym.Env):
    """
    The base class for all gordongames variants.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 targ_range=(1,10),
                 grid_size=(31,31),
                 pixel_density=5,
                 harsh=True,
                 *args, **kwargs):
        """
        Args:
            targ_range: tuple of ints (low, high) (both inclusive)
                the range of potential target counts for the game
            grid_size: tuple of ints (n_row, n_col)
                the dimensions of the grid in grid units
            pixel_density: int
                the number of pixels per unit in the grid
            harsh: bool
                changes the reward system to be more continuous if false
        """
        # determines the unit dimensions of the grid
        self.grid_size = grid_size
        # determines the number of pixels per grid unit
        self.pixel_density = pixel_density
        # tracks number of steps in episode
        self.step_count = 0
        # used in calculations of self.max_steps
        self.max_step_base = self.grid_size[0]//2*self.grid_size[1]*2
        # gets set in reset(), limits number of steps per episode
        self.max_steps = 0
        self.targ_range = targ_range
        if type(targ_range) == int:
            self.targ_range = (targ_range,targ_range)
        self.harsh = harsh
        self.viewer = None
        self.action_space = Discrete(6)
        self.is_grabbing = False
        self.set_controller()

    def set_controller(self):
        """
        Must override this function and set a member `self.controller`
        """
        self.controller = None # Must set a controller
        raise NotImplemented

    def _toggle_grab(self):
        grab = not self.is_grabbing
        coord = self.controller.register.player.coord
        if self.is_grabbing:
            self.is_grabbing = False
        # we know is_grabbing is false here
        elif not self.controller.register.is_empty(coord):
            self.is_grabbing = True
        return grab

    def step(self, action):
        """
        Args:
            action: int
                the action should be an int of either a direction or
                a grab command
                    0: null action
                    1: move up one unit
                    2: move right one unit
                    3: move down one unit
                    4: move left one unit
                    5: grab/drop object
        Returns:
            last_obs: ndarray
                the observation
            rew: float
                the reward
            done: bool
                if true, the episode has ended
            info: dict
                whatever information the game contains
        """
        self.step_count += 1
        if action < 5:
            direction = action
            grab = self.is_grabbing
        else:
            direction = STAY
            grab = self._toggle_grab()
        self.last_obs,rew,done,info = self.controller.step(
            direction,
            int(grab)
        )
        player = self.controller.register.player
        info["grab"] = str(self.get_other_obj(player, grab))
        if self.step_count > self.max_steps: done = True
        elif self.step_count == self.max_steps and rew == 0:
            rew = self.controller.max_punishment
            done = True
        return self.last_obs, rew, done, info

    def get_other_obj(self, obj, grab):
        """
        Finds and returns the first game object that is not the argued
        object and is located at the location of the argued object. 

        Args:
            obj: GameObject
            grab: bool
                the player's current grab state
        Returns:
            other_obj: GameObject or None
                one of the other objects located at this location.
                The priority goes by type:
                    Item
                    Player
                    Target
                    Pile
                    Button
        """
        if not grab: return None
        reg = self.controller.register.coord_register
        objs = {*reg[obj.coord]}
        if len(objs) == 1: return None
        objs.remove(obj)
        order = [ITEM, PLAYER, TARG, PILE, BUTTON]
        memo = {o: set() for o in order}
        # Sort objects
        for o in objs:
            memo[o.type].add(o)
        # return single object by priority
        for o in order:
            if len(memo[o]) > 0: return memo[o].pop()
        return None

    def reset(self, n_targs=None):
        self.controller.reset(n_targs=n_targs)
        self.max_steps = (self.controller.n_targs+1)*self.max_step_base
        self.is_grabbing = False
        self.step_count = 0
        self.last_obs = self.controller.grid.grid
        return self.last_obs

    def render(self, mode='human', close=False, frame_speed=.1):
        if self.viewer is None:
            self.fig = plt.figure()
            self.viewer = self.fig.add_subplot(111)
            plt.ion()
            self.fig.show()
        else:
            self.viewer.clear()
            self.viewer.imshow(self.last_obs)
            plt.pause(frame_speed)
        self.fig.canvas.draw()

    def seed(self, x):
        np.random.seed(x)

class EvenLineMatch(GordonGame):
    """
    Creates a gym version of Peter Gordon's Even Line Matching game.
    The user attempts to match the target object line within a maximum
    number of steps based on the size of the grid and the number of
    target objects on the grid. The maximum step count is enough so
    that the agent can walk around the perimeter of the playable area
    n_targs+1 number of times. The optimal policy will always be able
    to finish well before this.
    """
    def set_controller(self):
        self.controller = EvenLineMatchController(
            grid_size=self.grid_size,
            pixel_density=self.pixel_density,
            harsh=self.harsh,
            targ_range=self.targ_range
        )
        self.controller.reset()

class ClusterMatch(GordonGame):
    """
    Creates a gym version of Peter Gordon's Cluster Matching game.
    The user attempts to place the same number of items on the grid as
    the number of target objects. The target objects are randomly
    placed while the agent attempts to align the placed items along a
    single row.

    The number of steps is based on the size of the grid and the number
    of target objects on the grid. The maximum step count is enough so
    that the agent can walk around the perimeter of the playable area
    n_targs+1 number of times. The optimal policy will always be able
    to finish well before this.
    """
    def set_controller(self):
        self.controller = ClusterMatchController(
            grid_size=self.grid_size,
            pixel_density=self.pixel_density,
            harsh=self.harsh,
            targ_range=self.targ_range
        )
        self.controller.reset()

class OrthogonalLineMatch(GordonGame):
    """
    Creates a gym version of Peter Gordon's Orthogonal Line Matching
    game.  The user attempts to layout a number of items in a horizontal
    line to match the count of a number of target objects laying in a
    vertical line. It must do so within a maximum number of steps
    based on the size of the grid and the number of target objects on
    the grid. The maximum step count is enough so that the agent can
    walk around the perimeter of the playable area n_targs+1 number of
    times. The optimal policy will always be able to finish well
    before this.
    """
    def set_controller(self):
        self.controller = OrthogonalLineMatchController(
            grid_size=self.grid_size,
            pixel_density=self.pixel_density,
            harsh=self.harsh,
            targ_range=self.targ_range
        )
        self.controller.reset()

class UnevenLineMatch(GordonGame):
    """
    Creates a gym version of Peter Gordon's Uneven Line Matching game.
    The user attempts to match the target object line within a maximum
    number of steps based on the size of the grid and the number of
    target objects on the grid. The maximum step count is enough so
    that the agent can walk around the perimeter of the playable area
    n_targs+1 number of times. The optimal policy will always be able
    to finish well before this.
    """
    def set_controller(self):
        self.controller = UnevenLineMatchController(
            grid_size=self.grid_size,
            pixel_density=self.pixel_density,
            harsh=self.harsh,
            targ_range=self.targ_range
        )
        self.controller.reset()

class ReverseClusterMatch(GordonGame):
    """
    Creates a gym version of the reverse of Peter Gordon's Cluster
    Matching game. The user attempts to place the same number of items
    on the grid as the number of evenly spaced, aligned target objects.
    The placed items must not align with the target objects.

    The number of steps is based on the size of the grid and the number
    of target objects on the grid. The maximum step count is enough so
    that the agent can walk around the perimeter of the playable area
    n_targs+1 number of times. The optimal policy will always be able
    to finish well before this.
    """
    def set_controller(self):
        self.controller = ReverseClusterMatchController(
            grid_size=self.grid_size,
            pixel_density=self.pixel_density,
            harsh=self.harsh,
            targ_range=self.targ_range
        )
        self.controller.reset()

class ClusterClusterMatch(GordonGame):
    """
    Creates a gym game in which the user attempts to place the same
    number of items on the grid as the number of target objects.
    The target objects are randomly placed and no structure is imposed
    on the placement of the user's items.

    The number of steps is based on the size of the grid and the number
    of target objects on the grid. The maximum step count is enough so
    that the agent can walk around the perimeter of the playable area
    n_targs+1 number of times. The optimal policy will always be able
    to finish well before this.
    """
    def set_controller(self):
        self.controller = ClusterClusterMatchController(
            grid_size=self.grid_size,
            pixel_density=self.pixel_density,
            harsh=self.harsh,
            targ_range=self.targ_range
        )
        self.controller.reset()

