import datetime
import pathlib
import torch
from libultimate import Action, UltimateEnv
import cv2
import numpy as np
import os
from libultimate import Console, UltimateController, Action, UltimateEnv

from .abstract_game import AbstractGame

# 29 action
action_list = [
    Action.ACTION_JAB,
    Action.ACTION_RIGHT_TILT,
    Action.ACTION_LEFT_TILT,
    Action.ACTION_UP_TILT,
    Action.ACTION_DOWN_TILT,
    Action.ACTION_RIGHT_SMASH,
    Action.ACTION_LEFT_SMASH,
    Action.ACTION_UP_SMASH,
    Action.ACTION_DOWN_SMASH,
    Action.ACTION_NEUTRAL_SPECIAL,
    Action.ACTION_RIGHT_SPECIAL,
    Action.ACTION_LEFT_SPECIAL,
    Action.ACTION_UP_SPECIAL,
    Action.ACTION_UP_RIGHT_SPECIAL,
    Action.ACTION_UP_LEFT_SPECIAL,
    Action.ACTION_DOWN_SPECIAL,
    Action.ACTION_GRAB,
    Action.ACTION_SHIELD,
    Action.ACTION_JUMP,
    Action.ACTION_RIGHT_JUMP,
    Action.ACTION_LEFT_JUMP,
    Action.ACTION_SHORT_HOP,
    Action.ACTION_RIGHT_SHORT_HOP,
    Action.ACTION_LEFT_SHORT_HOP,
    #Action.ACTION_UP_TAUNT,
    #Action.ACTION_DOWN_TAUNT,
    #Action.ACTION_LEFT_TAUNT,
    #Action.ACTION_RIGHT_TAUNT,
    Action.ACTION_SPOT_DODGE,
    Action.ACTION_RIGHT_ROLL,
    Action.ACTION_LEFT_ROLL,
    #Action.ACTION_RIGHT_DASH,
    #Action.ACTION_LEFT_DASH,
    #Action.ACTION_RIGHT_WALK,
    #Action.ACTION_LEFT_WALK,
    #Action.ACTION_CROUCH,
    #Action.ACTION_RIGHT_CRAWL,
    #Action.ACTION_LEFT_CRAWL,
    Action.ACTION_RIGHT_STICK,
    Action.ACTION_LEFT_STICK,
    Action.ACTION_UP_STICK,
    Action.ACTION_DOWN_STICK,
    #Action.ACTION_NO_OPERATION
]

class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
        # channel: [1p, 2p], dim: (lr, percent, pos_x, pos_y, sti_x, sti_y, b_att, b_spe, b_sma, b_gur, b_gur_h, b_catch, b_jump, b_jump_mini)
        self.observation_shape = (2, 1, 14)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(len(action_list)))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = True
        self.max_moves = 2500  # Maximum number of moves if game is not finished before
        self.num_simulations = 30  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = "resnet"  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 2  # Number of blocks in the ResNet
        self.channels = 16  # Number of channels in the ResNet
        self.reduced_channels_reward = 4  # Number of channels in reward head
        self.reduced_channels_value = 4  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [16]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [16]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [16]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 10
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [16]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network

        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = int(1000e6)  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 16  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 500  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.005  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 350e3

        ### Replay Buffer
        self.replay_buffer_size = int(1e5)  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 5  # Number of game moves to keep for every batch element
        self.td_steps = 10  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 1  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = False  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 500e3:
            return 1.0
        elif trained_steps < 750e3:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        RYUJINX_PATH = os.path.join(os.path.dirname(__file__), "../libultimate/test")
        console = Console(ryujinx_path=RYUJINX_PATH)
        controller = UltimateController()
        self.env = UltimateEnv(console, controller, hz=6, action_list=action_list)

    def _convert_obs(self, gamestate):
        # channel: [1p, 2p], dim: (lr, percent, pos_x, pos_y, sti_x, sti_y, b_att, b_spe, b_sma, b_gur, b_gur_h, b_catch, b_jump, b_jump_mini)
        new_obs = np.zeros((2, 1, 14))
        for i, p in enumerate(gamestate.players):
            new_obs[i][0][0] = p.lr # left: -1, right: 1
            new_obs[i][0][1] = p.percent / 150 if p.percent < 150 else 1
            new_obs[i][0][2] = p.position.x / 200  # FD stage x is -200 ~ +200
            new_obs[i][0][3] = p.position.y / 150 # FD stage y is -150 ~ +150
            new_obs[i][0][4] = p.control_state.stick_x
            new_obs[i][0][5] = p.control_state.stick_y
            new_obs[i][0][6] = 1 if p.control_state.button_attack else 0
            new_obs[i][0][7] = 1 if p.control_state.button_special else 0
            new_obs[i][0][8] = 1 if p.control_state.button_smash else 0
            new_obs[i][0][9] = 1 if p.control_state.button_guard else 0
            new_obs[i][0][10] = 1 if p.control_state.button_guard_hold else 0
            new_obs[i][0][11] = 1 if p.control_state.button_catch else 0
            new_obs[i][0][12] = 1 if p.control_state.button_jump else 0
            new_obs[i][0][13] = 1 if p.control_state.button_jump_mini else 0
        return new_obs
    
    def step(self, action_num):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        action = action_list[action_num]
        obs, reward, done, info = self.env.step(action)
        # obs is gamestate
        obs = self._convert_obs(obs)
        # reward fix to 0-1
        reward = reward / 50 # 10% damage is 0.2 reward
        return obs, reward, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(len(action_list)))

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        obs = self.env.reset(without_reset=False)
        return obs

    def close(self):
        """
        Properly close the game.
        """
        #self.env.close()
        print("close")
        pass

    def render(self):
        """
        Display the game observation.
        """
        #self.env.render()
        input("Press enter to take a step ")

