import abc
import numpy as np # For type hinting numpy arrays

class BaseGameAI(abc.ABC):
    """
    Abstract Base Class for game environments to be used with the AI agent.
    All concrete game implementations (e.g., SnakeGameAI, PongGameAI)
    must inherit from this class and implement its abstract methods.
    """

    @abc.abstractmethod
    def reset(self) -> None:
        """Resets the game state to its initial configuration."""
        pass

    @abc.abstractmethod
    def get_state(self) -> np.ndarray:
        """
        Returns the current state of the game as a NumPy array.
        The format (e.g., feature vector, pixel array) depends on the game.
        """
        pass

    @abc.abstractmethod
    def play_step(self, action_index: int) -> tuple[float, bool, int]:
        """
        Performs one step in the game given an action.
        Args:
            action_index (int): An integer representing the chosen action.

        Returns:
            tuple[float, bool, int]: (reward, game_over, score)
        """
        pass

    @abc.abstractmethod
    def is_game_over(self) -> bool:
        """Checks if the game has ended."""
        pass

    @abc.abstractmethod
    def get_action_space_size(self) -> int:
        """Returns the number of possible discrete actions."""
        pass

    @abc.abstractmethod
    def get_state_shape(self) -> tuple[int, ...]:
        """
        Returns the shape of the game's state representation.
        For feature vectors: (num_features,)
        For image data: (channels, height, width)
        """
        pass

    @abc.abstractmethod
    def get_state_is_image(self) -> bool:
        """
        Returns True if the game's state representation is pixel-based (image),
        False if it's a feature vector.
        """
        pass

    @abc.abstractmethod
    def get_score(self) -> int:
        """Returns the current score of the agent in the game."""
        pass

    @abc.abstractmethod
    def render(self) -> None:
        """Updates the game's graphical user interface."""
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """Performs any necessary cleanup when the game is closed (e.g., pygame.quit())."""
        pass

    @abc.abstractmethod
    def set_speed(self, new_speed: int) -> None:
        """Sets the game's simulation speed (e.g., FPS)."""
        pass
