# PE50-AI-for-the-Game-of-GO ★ AlphaGo-Lite (9×9)

A compact, research-grade re-implementation of the **AlphaGo Zero** pipeline on a 9 × 9 board.  
The engine blends **Monte-Carlo Tree Search (MCTS)** with twin neural networks (policy + value).  
► **Bootstrap phase:** networks are first trained on thousands of **KataGo-generated games** (no human records).  
► **Improvement phase:** the agent continues to refine itself through iterative **self-play**. 

---

## ✨ Key Features

| Axis | Highlights |
|------|------------|
| **Search / MCTS** | PUCT selection, Dirichlet noise at the root, value rollouts replaced by NN evaluation. Default **2 500 sims** per move and **C<sub>PUCT</sub> = 1.1**  |
| **Policy Net** | 8-block ResNet with Squeeze-and-Excitation attention; dropout-optimised head; predicts 81 moves + pass in one softmax  |
| **Value Net** | 6 residual blocks, global attention and Sigmoid head yielding a win-probability in \[0–1]  |
| **Reinforcement Loop** | 50 self-play games → 25 training epochs → evaluation vs previous net every cycle; batching & multi-GPU friendly CLI  |
| **Dataset Bootstrap** | ≈30 k 9×9 games produced by **KataGo** (ELO ≈ 14 k) → 17 M positions after 8-fold symmetry augmentation (provides a strong starting point before self-play)  |
| **Interactive GUI** | Tkinter front-end (5×5 – 19×19), AI-vs-Human, AI-vs-AI, sandbox & heat-map overlay options  |

---

## 📂 Project Structure

```

PE50-AI-for-the-Game-of-GO/
├── main.py                       # Tkinter GUI & game launcher
├── MCTS\_GO.py                    # Core MCTS implementation
├── katago\_relatives/
│   └── katago-opencl-linux/
│       └── networks/
│           ├── go\_policy\_9x9\_v5.py           # Policy network
│           ├── go\_value\_9x9\_v3.py            # Value network
│           └── reinforcement\_learning.py     # Self-play + RL driver
├── plateau.py | noeud\_go.py | arbre\_go.py    # Board & tree helpers
├── moteursDeJeu.py | const.py                # Game engines & constants
├── ressource/                                # GUI assets (background etc.)
└── docs/plots/ (optional)                    # Training curves (add yours!)

````

---

## ⚙️ Installation

```bash
# Clone & enter
git clone https://github.com/Skyxo/PE50-AI-for-the-Game-of-GO.git
cd PE50-AI-for-the-Game-of-GO

# Create a virtual-env (Python ≥ 3.9 recommended)
python -m venv .venv && source .venv/bin/activate

# Core dependencies
pip install torch numpy matplotlib tqdm tkinter  # plus anything you add later
````

*KataGo binaries* for data boot-strapping are vendored under `katago_relatives/`; feel free to swap in a newer network.

---

## 🚀 Quick Start

### 1 ▪ Pre-Training + Self-Play

```bash
python katago_relatives/katago-opencl-linux/networks/reinforcement_learning.py \
  --iterations 100 \          # RL cycles
  --games 50 \                # self-play games / cycle
  --epochs 25 \               # NN epochs / cycle
  --mcts_sims 800 \           # simulations / move
  --eval_simulations 3200 \   # sims for evaluation
  --eval_games 20             # evaluation matches / cycle
```

*During the **first** cycle the script loads the KataGo-generated dataset and performs supervised pre-training on policy & value heads before switching to pure self-play.*
Resume any run with `--resume checkpoints/latest.pth`.

### 2 ▪ Play Through the GUI

```bash
python main.py
```

Inside the launcher you can:

* Choose board size (5 / 9 / 13 / 19)
* Launch **Sandbox**, **Human vs Human**, **Play vs AI**, or **AI vs AI**
* Toggle the heat-map overlay to visualise MCTS visits.

---

## 📊 Training Metrics

During training the script logs CSV and generates plots for:

* **Policy loss** & **Value loss** per epoch
* **Win-rate** against previous generations
* **Experience buffer size** growth

Add your graphs under `docs/plots/` and embed them here for a quick visual.

---

## 🔬 Research Internals

* **Minimal input planes** – only current & opponent stones: forces the net to learn tactics, not hand-coded patterns.
* **Top-k expansion** – keeps the 15 best moves (9×9) when expanding a node, capping tree width.
* **Model refresh** – periodic “weight-re-load” halves optimiser momentum to escape plateaus.

---

## 🛣 Roadmap

* [ ] Curriculum learning for 13×13 then 19×19
* [ ] SGF export + GTP bridge
* [ ] Distributed self-play (Ray / MPI)
* [ ] Heat-map overlay inside the GUI
* [ ] CI pipeline & pre-commit hooks

---

## 🤝 Contributing

1. **Fork** ➜ create a feature branch (`feat/your-feature`)
2. Format with `black` / `flake8`, run tests if you add them
3. Open a well-described Pull Request (screenshots welcome!)

---

## 📜 Credits & Licence

*Built by the PE-50 team (École Centrale de Lyon, 2025).*
Inspired by **DeepMind’s AlphaGo Zero** and the open-source **KataGo** community.
Released under the **MIT Licence** – see `LICENSE` for details.
