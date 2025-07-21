# PE50-AI-for-the-Game-of-GO ★ AlphaGo-Lite (9 × 9)

A compact, research-grade re-implementation of the **AlphaGo Zero** pipeline on a 9 × 9 board.

📄 **Project report (PDF)** → [docs/Rapport_PE_050_2024.pdf](docs/Rapport_PE_050_2024.pdf)  
*Section 5 (Neural Networks) and all technical appendices were written by **Charles Bergeat***.

The engine combines **Monte-Carlo Tree Search (MCTS)** with twin neural networks (policy + value).  
- **Bootstrap phase :** the networks are first trained on **30 000 KataGo-generated games** (no human records).  
- **Improvement phase :** the agent then refines itself through iterative **self-play**.

---

### 🎮 A Glimpse of the GUI

| Main launcher (board size & mode) | In-game 19 × 19 goban |
|---|---|
| <img src="https://github.com/user-attachments/assets/2aea3f62-bd35-46dd-abd2-c2f29c8ba73d" width="370" alt="Main menu"> | <img src="https://github.com/user-attachments/assets/da6dc5ab-a4ef-4c41-a0d1-904807f05c49" width="370" alt="19x19 board"> |

*From the menu you can spawn a **sandbox**, a **local 2-player** match, play vs the AI (black / white) or run **AI vs AI** benchmarks on 5×5, 9×9, 13×13 or 19×19 boards.*

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

## 📊 Training & Evaluation Results

| **Neural agent vs. pure MCTS** (100-game match-up) |
|----------------------------------------------------|
| <img src="https://github.com/user-attachments/assets/a12eb50e-accd-4b43-becf-45df975961dc" width="650" alt="Win-rate histogram"> |

Our **NN-guided MCTS** (green) wins **≈ 96 %** of games against a vanilla, heuristic-only MCTS baseline.

### 1 · Value-Net learning curve

<img src="https://github.com/user-attachments/assets/5550562e-4ab5-4471-b521-850d9a8003e5" width="720" alt="Value-net loss & R²">

* **Top** – training (blue) vs validation (red) *MSE* steadily declines; periodic “refresh” events (green dashed) lower the LR and reset optimiser momentum.  
* **Bottom** – coefficient of determination **R²** climbs from 0.58 → 0.75, then plateaus once the network converges.

### 2 · Policy-Net entropy

<img src="https://github.com/user-attachments/assets/d438a4ef-0da5-4350-abb1-64191c79b077" width="720" alt="Policy entropy">

Prediction entropy falls from ~3.4 nats to ~2.1 nats, indicating the policy becomes more confident while still preserving exploration; target-distribution switches and optimizer refreshes are marked for reference.

### 3 · Losses across RL iterations

<img src="https://github.com/user-attachments/assets/e53fb909-7ab4-42e6-863c-4b0a7056e906" width="720" alt="Policy & Value losses by iteration">

Each colour tracks one reinforcement-learning iteration (25 epochs).  
* **Left** – policy KL-divergence shrinks iteration after iteration.  
* **Right** – value-net R² improves in lock-step, reflecting tighter win-probability estimates.

---

**Key takeaways**

* Supervised bootstrap + RL cuts **policy KL** from 3.4 → 1.9 and boosts **value R²** to **0.75**.  
* Scheduled **optimizer refresh** keeps both nets from stagnating after LR drops.  
* Resulting agent achieves **strong generalisation** — high win-rate on previously unseen openings & vs baseline bots.

> Feel free to add your own figures (PNG/SVG) under `docs/plots/` and embed them here with a short caption — consistency beats quantity!


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




