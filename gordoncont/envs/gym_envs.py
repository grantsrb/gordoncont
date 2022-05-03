import os, subprocess, time, signal
import gym
from gordoncont.ggames import Discrete, Box
from gordoncont.ggames.controllers import *
from gordoncont.ggames.constants import STAY, ITEM, TARG, PLAYER, PILE, BUTTON, OBJECT_TYPES
from gordoncont.ggames.utils import find_empty_space_along_row
import numpy as np
import time

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
                 egocentric=False,
                 hold_outs=set(),
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
            egocentric: bool
                determines if perspective of the game will be centered
                on the player. If true, the perspective is centered on
                the player. It is important to note that the
                observations double in size to ensure that all info in
                the game is always accessible.
            hold_outs: set of ints
                a set of integer values representing numbers of targets
                that should not be sampled when sampling targets
        """
        self.egocentric = egocentric
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
        self.max_items = self.targ_range[-1]*3
        self.harsh = harsh
        self.hold_outs = hold_outs
        self.viewer = None
        self.action_space = Box((3,))
        self.is_grabbing = False
        self.seed(int(time.time()))
        self.set_controller()

    def set_controller(self):
        """
        Must override this function and set a member `self.controller`
        """
        self.controller = None # Must set a controller
        raise NotImplemented

    def _toggle_grab(self):
        """
        Toggles the grab state of the player. If the task is either the
        brief presentation or nuts in can task, the grabbing is
        restricted until the initial animations have finished.
        """
        grab = not self.is_grabbing
        coord = self.controller.register.player.coord
        if self.is_grabbing:
            self.is_grabbing = False
        # we know is_grabbing is currently false and there is an object
        # under the player
        elif not self.controller.register.is_empty(coord):
            self.is_grabbing = True
        # Restrict grabbing if experiencing initial animations in
        # BriefPresentation or NutsInCan tasks.
        if type(self.controller)==BriefPresentationController and\
                self.controller.is_animating:
            self.is_grabbing = False
            grab = False
        return grab

    def step(self, action):
        """
        Args:
            action: tuple of floats (3,) -> (x, y, grab)
                the action should be a tuple of floats with the first
                two indices indicating the x,y coordinate and the last
                index indicating whether or not to grab
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
        xycoord = action[:2]
        grab = int(action[2]>0)
        self.last_obs,rew,done,info = self.controller.step(
            xycoord,
            grab
        )
        player = self.controller.register.player
        info["grab"] = self.get_other_obj_idx(player, grab)
        if self.step_count > self.max_steps: done = True
        if info["n_items"] > self.max_items: done = True
        elif self.step_count == self.max_steps and rew == 0:
            rew = self.controller.max_punishment
            done = True
        return self.last_obs, rew, done, info

    def get_other_obj_idx(self, obj, grab):
        """
        Finds and returns an int representing the first game object
        that is not the argued object and is located at the locations
        of the argued object. The priority of objects is detailed
        by the priority list.

        Args:
            obj: GameObject
            grab: bool
                the player's current grab state
        Returns:
            other_obj: GameObject or None
                one of the other objects located at this location.
                The priority goes by type, see `priority`. a return
                of 0 means the player is either not grabbing or there
                are no items to grab
        """
        # Langpractice depends on this order
        keys = sorted(list(PRIORITY2TYPE.keys()))
        priority = [ PRIORITY2TYPE[k] for k in keys ]
        if not grab: return 0
        reg = self.controller.register.coord_register
        objs = {*reg[obj.coord]}
        if len(objs) == 1: return 0
        objs.remove(obj)
        memo = {o: set() for o in priority}
        # Sort objects
        for o in objs:
            memo[o.type].add(o)
        # return single object by priority
        for o in priority:
            if len(memo[o]) > 0: return TYPE2PRIORITY[o]
        return 0

    def reset(self, n_targs=None, *args, **kwargs):
        self.controller.rand = self.rand
        self.controller.reset(n_targs=n_targs)
        self.max_steps = (self.controller.n_targs+1)*self.max_step_base
        self.is_grabbing = False
        self.step_count = 0
        coord = self.controller.register.player.coord
        self.last_obs = self.controller.grid.get_grid(coord)
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
        self.rand = np.random.default_rng(x)

    def set_seed(self, x):
        self.rand = np.random.default_rng(x)
        pass

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
            targ_range=self.targ_range,
            egocentric=self.egocentric,
            hold_outs=self.hold_outs
        )
        self.controller.rand = self.rand
        self.controller.reset()

class ClusterMatch(GordonGame):
    """
    Creates a gym version of Peter Gordon's Cluster Matching game.
    The user attempts to place the same number of items on the grid as
    the number of target objects. The target objects are randomly
    placed while the agent attempts to align the placed items along a
    single row.
    """
    def set_controller(self):
        self.controller = ClusterMatchController(
            grid_size=self.grid_size,
            pixel_density=self.pixel_density,
            harsh=self.harsh,
            targ_range=self.targ_range,
            egocentric=self.egocentric,
            hold_outs=self.hold_outs
        )
        self.controller.rand = self.rand
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
            targ_range=self.targ_range,
            egocentric=self.egocentric,
            hold_outs=self.hold_outs
        )
        self.controller.rand = self.rand
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
            targ_range=self.targ_range,
            egocentric=self.egocentric,
            hold_outs=self.hold_outs
        )
        self.controller.rand = self.rand
        self.controller.reset()

class ReverseClusterMatch(GordonGame):
    """
    Creates a gym version of the reverse of Peter Gordon's Cluster
    Matching game. The user attempts to place the same number of items
    on the grid as the number of evenly spaced, aligned target objects.
    The placed items must not align with the target objects.
    """
    def set_controller(self):
        self.controller = ReverseClusterMatchController(
            grid_size=self.grid_size,
            pixel_density=self.pixel_density,
            harsh=self.harsh,
            targ_range=self.targ_range,
            egocentric=self.egocentric,
            hold_outs=self.hold_outs
        )
        self.controller.rand = self.rand
        self.controller.reset()

class ClusterClusterMatch(GordonGame):
    """
    Creates a gym game in which the user attempts to place the same
    number of items on the grid as the number of target objects.
    The target objects are randomly placed and no structure is imposed
    on the placement of the user's items.
    """
    def set_controller(self):
        self.controller = ClusterClusterMatchController(
            grid_size=self.grid_size,
            pixel_density=self.pixel_density,
            harsh=self.harsh,
            targ_range=self.targ_range,
            egocentric=self.egocentric,
            hold_outs=self.hold_outs
        )
        self.controller.rand = self.rand
        self.controller.reset()

class BriefPresentation(GordonGame):
    """
    Creates a gym game in which the user attempts to place the same
    number of items on the grid as the number of target objects.
    The target objects are randomly placed and the agent is supposed
    to place the same number of items aligned along a single row. The
    agent's movement is restricted for the first DISPLAY_COUNT frames.
    The targets are removed from the agent's visual display after the
    DISPLAY_COUNT frames and the agent has to perform the counting task
    from memory.
    """
    def set_controller(self):
        self.controller = BriefPresentationController(
            grid_size=self.grid_size,
            pixel_density=self.pixel_density,
            harsh=self.harsh,
            targ_range=self.targ_range,
            egocentric=self.egocentric,
            hold_outs=self.hold_outs
        )
        self.controller.rand = self.rand
        self.controller.reset()

class NutsInCan(GordonGame):
    """
    Creates a gym version of Peter Gordon's Nuts-In-A-Can game.

    This class creates a game in which the environment initially flashes
    the targets one by one until all targets are flashed. At the end
    of the flashing, a center piece appears (to indicate that the
    flashing stage is over). The agent must then place the same number
    of items as there are targets (each of which was flashed only
    briefly at the beginning of the game).

    Once the agent believes the number of items is equal to the number
    of targets, they must press the ending button.
    """
    def set_controller(self):
        self.controller = NutsInCanController(
            grid_size=self.grid_size,
            pixel_density=self.pixel_density,
            harsh=self.harsh,
            targ_range=self.targ_range,
            egocentric=self.egocentric,
            hold_outs=self.hold_outs
        )
        self.controller.rand = self.rand
        self.controller.reset()

class VisNuts(GordonGame):
    """
    Creates a gym version of Peter Gordon's Nuts-In-A-Can game in which
    the nuts remain visible.

    This class creates a game in which the environment has an initial
    animation in which the targets are flashed one by one until all
    targets are visible. At the end of the animation, a center piece
    appears (as an indication that the flashing stage is over).
    The agent must then place the same number of items as there are
    targets.
    """
    def set_controller(self):
        self.controller = VisNutsController(
            grid_size=self.grid_size,
            pixel_density=self.pixel_density,
            harsh=self.harsh,
            targ_range=self.targ_range,
            egocentric=self.egocentric,
            hold_outs=self.hold_outs
        )
        self.controller.rand = self.rand
        self.controller.reset()

