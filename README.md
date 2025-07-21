# PE50-AI-for-the-Game-of-GO ★ AlphaGo-Lite (9 × 9)

A compact, research-grade re-implementation of the **AlphaGo Zero** pipeline on a 9 × 9 board.

📄 **Project report (PDF)** → [docs/Rapport_PE_050_2024.pdf](docs/Rapport_PE_050_2024.pdf)  
*Section 5 (Neural Networks) and all technical appendices were written by **_Your Name_***.

The engine combines **Monte-Carlo Tree Search (MCTS)** with twin neural networks (policy + value).  
- **Bootstrap phase :** the networks are first trained on **30 000 KataGo-generated games** (no human records).  
- **Improvement phase :** the agent then refines itself through iterative **self-play**.

---

## ✨ Key Features

| Axis                | Highlights                                                                                                                                                          |
|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Search / MCTS**   | PUCT selection, Dirichlet noise at the root, NN value evaluation. Default **2 500 sims** per move with **C<sub>PUCT</sub> = 1.1**.                                   |
| **Policy Network**  | 8-block ResNet with Squeeze-and-Excitation; dropout-optimised head; predicts 81 moves + pass in a single softmax.                                                    |
| **Value Network**   | 6 residual blocks with global attention; Sigmoid head yielding a win-probability in \[0–1].                                                                         |
| **Reinforcement Loop** | 50 self-play games → 25 training epochs → evaluation vs. previous net every cycle; CLI supports batching & multi-GPU.                                            |
| **Dataset Bootstrap** | 30 k 9×9 games produced by **KataGo** (ELO ≈ 14 k) → 17 M positions after 8-fold symmetry augmentation, giving a strong starting point.                            |
| **Interactive GUI** | Tkinter front-end (5×5 – 19×19): AI-vs-Human, AI-vs-AI, sandbox mode & optional heat-map overlay.                                                                    |

---

## 📂 Project Structure

```text
PE50-AI-for-the-Game-of-GO/
├── main.py                           # Tkinter GUI & game launcher
├── MCTS_GO.py                        # Core MCTS implementation
├── katago_relatives/
│   └── katago-opencl-linux/
│       └── networks/
│           ├── go_policy_9x9_v5.py   # Policy network
│           ├── go_value_9x9_v3.py    # Value network
│           └── reinforcement_learning.py  # Self-play + RL driver
├── plateau.py | noeud_go.py | arbre_go.py  # Board & tree helpers
├── moteursDeJeu.py | const.py              # Game engines & constants
├── ressource/                              # GUI assets
└── docs/
    ├── Rapport_PE_050_2024.pdf             # Project report (this PDF)
    └── plots/                              # Training curves
````

---

## ⚙️ Installation

```bash
# Clone & enter
git clone https://github.com/Skyxo/PE50-AI-for-the-Game-of-GO.git
cd PE50-AI-for-the-Game-of-GO

# Create a virtual env (Python ≥ 3.9 recommended)
python -m venv .venv && source .venv/bin/activate

# Core dependencies
pip install torch numpy matplotlib tqdm tkinter
```

*KataGo binaries* for data boot-strapping are vendored in `katago_relatives/`; you can swap in a newer network if desired.

---

## 🚀 Quick Start

### 1 · Pre-Training + Self-Play

```bash
python katago_relatives/katago-opencl-linux/networks/reinforcement_learning.py \
  --iterations 100 \         # RL cycles
  --games 50 \               # self-play games per cycle
  --epochs 25 \              # NN epochs per cycle
  --mcts_sims 800 \          # simulations per move
  --eval_simulations 3200 \  # sims for evaluation
  --eval_games 20            # evaluation matches per cycle
```

During the **first** cycle the script loads the KataGo dataset and performs supervised pre-training on policy & value heads before switching to pure self-play.
Resume a run with `--resume checkpoints/latest.pth`.

### 2 · Play Through the GUI

```bash
python main.py
```

* Choose board size (5 / 9 / 13 / 19).
* Launch **Sandbox**, **Human vs Human**, **Play vs AI**, or **AI vs AI**.
* Toggle the heat-map overlay to visualise MCTS visits.

---

## 📊 Training Metrics

During training the script logs CSV files and can generate plots for :

* **Policy loss** & **Value loss** per epoch
* **Win-rate** against previous generations
* **Experience buffer size** growth

Add your graphs to `docs/plots/` and embed them here for a quick visual.

---

## 🔬 Research Internals

* **Minimal input planes** – only current & opponent stones: forces the net to learn tactics, not hand-crafted patterns.
* **Top-k expansion** – keeps the 15 best moves (9×9) when expanding a node, capping tree width.
* **Model refresh** – periodic “weight-reload” halves optimiser momentum to escape plateaus.

---

## 🛣️ Roadmap

* [ ] Curriculum learning for 13×13 then 19×19
* [ ] SGF export + GTP bridge
* [ ] Distributed self-play (Ray / MPI)
* [ ] Heat-map overlay inside the GUI
* [ ] CI pipeline & pre-commit hooks

---

## 🤝 Contributing

1. **Fork** → create a feature branch (`feat/your-feature`).
2. Format with `black` / `flake8`, run tests if you add them.
3. Open a detailed Pull Request (screenshots welcome!).

---

## 📜 Credits & Licence

*Built by the PE-50 team (École Centrale de Lyon, 2025).*
Inspired by **DeepMind’s AlphaGo Zero** and the open-source **KataGo** community.
Code released under the **MIT Licence** — see `LICENSE` for details.
