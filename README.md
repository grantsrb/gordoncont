# Gordon Games Continuous

## Description
gordongames continuous (gordoncont) is a continuous gym environment for recreating computational versions of games proposed in Peter Gordon's paper [_Numerical Cognition Without Words: Evidence from Amazonia_](https://www.science.org/doi/10.1126/science.1094492). 

## Dependencies
- python3
- pip
- gym
- numpy
- matplotlib

## Installation
1. Clone this repository
2. Navigate to the cloned repository
3. Run command `$ pip install -e ./`
4. add `import gordoncont` to the top of your python script
5. make one of the envs with the folowing: `env = gym.make("gordoncont-<version here>")`

## Rendering
A common error about matplotlib using `agg` can be fixed by including the following lines in your scripts before calling `.render()`:

    import matplotlib
    matplotlib.use('TkAgg')

If you are experiencing trouble using the `render()` function while using jupyter notebook, insert:

    %matplotlib notebook

before calling `render()`.

## Using gordoncont
After installation, you can use gordoncont by making one of the gym environments. See the paper [_Numerical Cognition Without Words: Evidence from Amazonia_](https://www.science.org/doi/10.1126/science.1094492) for more details about each game.

#### Environment v0 Even Line Match
Use `gym.make('gordoncont-v0')` to create the Line Match game. The agent must match the number of target objects by aligning them within the target columns. Targets are evenly spaced. These are the default options for the game (see Game Details to understand what each variable does):

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)
    egocentric = False

#### Environment v1 Cluster Match
Use `gym.make('gordoncont-v1')` to create the Cluster Line Match game. The agent must match the number target objects, but the target objects are randomly distributed and the agent must align the items in a row. These are the default options for the game (see Game Details to understand what each variable does):

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)
    egocentric = False

#### Environment v2 Orthogonal Line Match
Use `gym.make('gordoncont-v2')` to create the Orthogonal Line Match game. The agent must match the number of target objects, but the target objects are aligned vertically whereas the agent must align the items along a single row. These are the default options for the game (see Game Details to understand what each variable does):

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)
    egocentric = False

#### Environment v3 Uneven Line Match
Use `gym.make('gordoncont-v3')` to create the Uneven Line Match game. The agent must match the target objects by aligning them along each respective target column. The targets are unevenly spaced. These are the default options for the game (see Game Details to understand what each variable does):

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)
    egocentric = False

#### Environment v4 Nuts-In-Can
Use `gym.make('gordoncont-v4')` to create the Nuts-In-Can game. The agent initially watches a number of target objects get breifly flashed, one-by-one. These targets are randomly distributed about the target area. After the initial flash, each target is no longer visible. After all targets are flashed, the agent must then grab the pile the same number of times as there are targets. These are the default options for the game (see Game Details to understand what each variable does):

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)
    egocentric = False

#### Environment v5 Reverse Cluster Match
Use `gym.make('gordoncont-v5')` to create the Reverse Cluster Line Match game. The agent must match the number of target objects without aligning them. These are the default options for the game (see Game Details to understand what each variable does):

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)
    egocentric = False

#### Environment v6 Cluster Cluster Match
Use `gym.make('gordoncont-v6')` to create the Cluster Cluster Match game. The target objects are distributed randomly. The agent must simply match the number of target objects with no structure imposed. These are the default options for the game (see Game Details to understand what each variable does):

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)
    egocentric = False

#### Environment v7 Brief Display
Use `gym.make('gordoncont-v7')` to create the Brief Display game. This is the same as the Cluster Match variant except that the targets are only displayed for the first few frames of the game. The agent must then match the number of randomly distributed target objects from memory. 

These are the default options for the game (see Game Details to understand what each variable does):

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)
    egocentric = False

#### Environment v8 Visible Nuts-In-Can
Use `gym.make('gordoncont-v8')` to create the Visible Nuts-in-Can game. This is the same as the Nuts-In-Can variant except that the targets are displayed for the entire episode. 

These are the default options for the game (see Game Details to understand what each variable does):

    grid_size = [33,33]
    pixel_density = 1
    targ_range = (1,10)
    egocentric = False

## Game Details
Each game consists of a randomly intitialized grid with various objects distributed on the grid depending on the game type. The goal is for the agent to first complete some task and then press the end button located in the upper right corner of the grid. Episodes last until the agent presses the end button. The agent can move left, up, right, down, or stay still. The agent also has the ability to interact with objects via the grab action. Grab only acts on objects in the same square as the agent. If the object is an "item", the agent carries the item to wherever it moves on that step. If the object is a "pile", a new item is created and carried with the agent for that step. The ending button is pressed using the grab action. The reward is only granted at the end of each episode if the task was completed successfully.

#### Rewards
A +1 reward is returned only in the event of a successful completion of the task.

A -1 reward is returned when a task ends unsuccessfully.

##### Environment v0
The agent receives a +1 reward if each target has a single item located in its column at the end of the episode.

##### Environment v1
The agent receives a +1 reward if there exists a single item for each target. The agent must align the items along a single row.

##### Environment v2
The agent receives a +1 reward if there exists an item for each target. All items must be aligned along a single row.

##### Environment v3
The agent receives a +1 reward if each target has a single item located in its column at the end of the episode.

##### Environment v4
The agent receives a +1 reward if the agent removes the exact number of items placed in the pile.

##### Environment v5
The agent receives a +1 reward if there exists an item for each target. All items must not be aligned with the target objects.

##### Environment v6
The agent receives a +1 reward if there exists an item for each target.

##### Environment v7
The agent receives a +1 reward if there exists a single item for each target. The agent must align the items along a single row.

##### Environment v8
The agent receives a +1 reward if the agent removes the exact number of items placed in the pile.

#### Game Options

- `grid_size` - An row,col coordinate denoting the number of units on the grid (height, width).
- `pixel_density` - Number of numpy pixels within a single grid unit.
- `targ_range` - A range of possible initial target object counts for each game (inclusive). Must be less than `grid_size`. 
- `egocentric` - If true, the perspective of the game will always be centered on the player. Be warned that this will double
                the size of the grid so as to maintain full information at every step of the game.

Each of these options are member variables of the environment and will come into effect after the environment is reset. For example, if you wanted to use 1-5 targets in game A, you can be set this using the following code:

    env = gym.snake('gordoncont-v0')
    env.targ_range = (1,5)
    observation = env.reset()
    ...
    # You can specify the number of targets directly at reset
    observation = env.reset( n_targs=5 )


#### Environment Parameter Examples
Examples coming soon!

#### About the Code
Coming soon!
