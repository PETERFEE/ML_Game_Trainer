# ML Game Trainer

A machine learning game training platform with multiple classic games and AI agents.

## Features

- **Multiple Games**: Snake, Pong, Flappy Bird, Tetris, Galaga, Dino Run, Packman
- **AI Training**: Deep Q-Learning with neural networks
- **Real-time Visualization**: Live training plots and game windows
- **Speed Control**: Adjustable game speed during training
- **Model Persistence**: Automatic model saving and loading

## Games

### Snake Game
- Classic snake gameplay with food collection
- Neural network learns optimal pathfinding

### Pong Game
- Two-player pong with AI paddle
- Improved reward structure and state representation

### Flappy Bird
- Bird navigation through pipes
- Continuous flight mechanics

### Tetris
- Block stacking and line clearing
- State includes current piece and board layout

### Galaga
- Space shooter with enemy waves
- Multiple enemy types and movement patterns

### Dino Run
- Endless runner with obstacles
- Random obstacle generation

### Packman
- Classic maze navigation with 3 ghosts
- Different ghost behaviors (Red: aggressive, Pink: conditional, Blue: random)
- 1/3 random bean placement

## Installation

```bash
pip install pygame torch numpy matplotlib tkinter
```

## Usage

```bash
python main.py
```

1. Select a game from the UI
2. Watch the AI train in real-time
3. Adjust speed and controls as needed
4. Models are automatically saved to `model/` directory

## Architecture

- **main.py**: Application entry point and game selection
- **ui_windows.py**: Tkinter UI for game selection and controls
- **game_runner.py**: Game loop and AI training coordination
- **agent_core.py**: Neural network agent implementation
- **game_interface.py**: Base class for all games
- **utils.py**: Utility functions for resource paths

## Model Files

Trained models are saved as:
- `model/model_[GameName]_[linear/cnn].pth`

## Requirements

- Python 3.7+
- Pygame
- PyTorch
- NumPy
- Matplotlib
- Tkinter 