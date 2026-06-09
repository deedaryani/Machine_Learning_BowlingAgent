# Unity ML-Agents: Intelligent Bowling Agent

This repository contains the reinforcement learning framework and trained models for an autonomous **Bowling Agent** developed using Unity ML-Agents and PyTorch. The project focuses on training an agent to dynamically adjust its position, launch velocity, and ball spin to maximise the number of pins knocked down per frame.

---

## 🚀 Project Overview

The core objective of this project is to train a virtual bowling opponent (`BowlingAgent`) using **Proximal Policy Optimization (PPO)**. The agent iteratively learns complex physics relationships—such as ball weight, lane friction, and rotational spin—by interacting with the simulated environment and receiving mathematical rewards based on its performance.

### Repository Context
* **Current Run Config:** `bowling_run_401`
* **Agent Context:** University of Liverpool: COMP 342 - Assignment 2

---

## 📂 Project Directory Structure

Below is the layout of the project workspace. The training outputs, configuration files, and core scripts are as follows:

```text
📁 201625668 Comp 342 Assignment 2/
└── 📁 bowling_run_401/
    ├── 📄 configuration.yaml                     # Hyperparameters for the PPO 
                                                    training run
    ├── 📄 BowlingAgent.onnx                      # Compiled deployment model for 
                                                    the Unity Inspector
    ├── 📁 BowlingAgent/                          # Main development outcomes
        ├── 📄BowlingAgent-50004.onnx             # Satisfactory outcomes with 
                                                    50,004 runs
        ├── 📄BowlingAgent-50004.pt               # PyTorch file that catches the    
                                                    characteristics of the agent
        ├── 📄checkpoint.pt                       # Checkpoint that is readable over 
                                                    terminal
    ├── 📁 run_logs
        ├── 📄timers.json                         # automatic receipt when the run 
                                                    ends
        ├── 📄training_status.json                # automatic receipt when the run 
                                                    ends
└── 📁 Game Executable
    ├── 📄 BowlingML.exe                          # executable file made in unity
└── 📁 Unity Project
    ├── 📁 Bowling ML
        ├── 📁 Assets
            ├── 📁 Scenes
                ├── 📄 Bowl.unity                 # Main unity file
                ├── 📄 Menu.unity                 # A simple Menu
                ├── 📁 Scripts
                    ├── 📄BowlingAgent.cs         # Agent Behaviour
                    ├── 📄BowlingController.cs    # Ball controller for setup
                    ├── 📄MainMenuController      # To prompt the main menu
                    ├── 📄TestBallShooter         # To test
└── 📄bowling_config.yaml                         # training parameters
└── 📄 README.md                                  # This project documentation file
└── 📄 BowlingAgent.onnx.png                      # Parameter Graph
    
```

---

## 🧠 Brain Architecture & Neural Network Profile

Based on programmatic extraction of the active model checkpoint (`.pt` / `.onnx`), the underlying brain topology functions as follows:


| Metric / Space | Config Size | Functional Mapping in Environment |
| :--- | :---: | :--- |
| **Vector Observations** | **2** | Tracks ball-to-pin alignment offset and relative trajectory distance. |
| **Continuous Actions** | **2** | Controls forward throw velocity (force) and physical hook/spin torque. |
| **Discrete Actions** | **1** | Selects baseline starting position lane profiles. |
| **Network Width** | **128** | Multi-Layer Perceptron (MLP) hidden layer neuron resolution. |

### Current Training Metrics (Checkpoint: 50,005 Steps)
* **Exploration Threshold (`log_sigma`):** `[[-0.0047, 0.0054]]`
  * *Status:* Mathematical deviation (e⁰ ≈ 1) confirms the agent is in a high-exploration phase, actively testing novel launch constraints to optimize its policy.
* **Observation Variance:** `[4533.98, 8322.27]`
  * *Status:* High raw numeric ranges are automatically tracked and standardized dynamically to prevent gradient explosions during real-time updates.

---

## 🛠️ Environmental Setup & Dependencies

To execute, debug, or continue training this model, ensure your system profile is configured with the following dependencies.

### Prerequisites
* Unity Editor (Recommended: 2022.3 LTS or newer)
* Unity ML-Agents Extensions Package
* Python 3.10+ / 3.11+

### Installation Flow
From your terminal window (PowerShell or Bash), run:

```powershell
# 1. Clone or navigate to the workspace folder
cd "D:\201625668 Comp 342 Assignment 2\bowling_run_401\BowlingAgent"

# 2. Install the necessary machine learning engines
pip install torch onnxruntime numpy
```

---

## 💻 How to Read and Execute the Model

### 1. Interrogating Checkpoints via Python
To quickly print out tensor layers, normalizer metadata, or network metrics directly from the PyTorch backend, run:

```python
import torch

# Load the target network checkpoint safely on CPU
checkpoint = torch.load("BowlingAgent-50004.pt", map_location=torch.device('cpu'))
policy_data = checkpoint['Policy']

print("=== Model Structural Bounds ===")
for key, value in policy_data.items():
    if "weight" not in key and "bias" not in key:
        print(f"{key}: {value.tolist() if hasattr(value, 'tolist') else value}")
```

### 2. Resuming the Training Vector
To advance the model out of its high-exploration baseline phase and into strict exploitation mode, execute the standard ML-Agents training command in your environment terminal:

```bash
mlagents-learn config/trainer_config.yaml --run-id=bowling_run_401 --resume
```

---

## 🎯 Reward Structure Map (Policy Alignment)

The policy optimizes its path calculation based on the following behavioral constraints:
* **Positive Reward (+1.0 per Pin):** Awarded incrementally for every structural pin knocked out of the active matrix space.
* **Strike Bonus (+5.0):** Immediate terminal bonus if all 10 pins drop on step 1.
* **Gutter Penalty (-2.0):** Applied instantly if the ball exits the lane boundary parameters before intersecting the pin coordinate grid.

## ✅ Comments
* **Behavioural Satisfaction:** Its a trainig agent which is still figuring out the best way to beat the game. It could be a good method to train opponent AI.
* **Note:** The Pins are inverted in the test run.