# AlphaGo-Lite ★ Reinforcement Learning for 9×9 Go

A compact re-implementation of the **AlphaGo Zero** pipeline on a 9×9 board.  
The engine couples **Monte-Carlo Tree Search (MCTS)** with twin neural networks (policy + value) and learns exclusively from **self-play**, needing **no human data**.

---

## ✨ Key Features

| Axis | Highlights |
|------|------------|
| **Search** | MCTS with **PUCT** selection, Dirichlet noise for exploration and value rollouts replaced by a NN evaluation :contentReference[oaicite:12]{index=12} |
| **Policy Net** | 8-block ResNet + Squeeze-and-Excitation, outputs full move distribution (81 + pass) in a single softmax :contentReference[oaicite:13]{index=13} |
| **Value Net** | 6 residual blocks with global attention, predicts win-probability in \[0, 1\] :contentReference[oaicite:14]{index=14} |
| **Dataset bootstrap** | 30 k 9×9 games auto-annotated with **KataGo** (ELO ≈ 14 k) → >17 M positions after 8-fold symmetry augmentation :contentReference[oaicite:15]{index=15} |
| **Self-play loop** | 50 games → 25 training epochs → 20 evaluation games per iteration; automatic promotion when win-rate > 50 % :contentReference[oaicite:16]{index=16} |
| **Fast progression** | First 3 iterations jump from 60 % to 73 % win-rate vs previous nets before plateauing :contentReference[oaicite:17]{index=17} |
| **GUI** | Tkinter board (5×5 – 19×19) for human play, AI vs AI or sandbox experimentation :contentReference[oaicite:18]{index=18} |

---

## 📂 Project Structure

```

alpha\_go\_lite/
├── MCTS\_GO.py                 # Core MCTS implementation
├── go\_policy\_9x9\_v5.py        # Policy network definition & training helpers
├── go\_value\_9x9\_v3.py         # Value network definition & training helpers
├── katago\_relatives/
│   └── katago-opencl-linux/
│       └── networks/
│           └── reinforcement\_learning.py  # Self-play + RL driver
├── gui/                        # Tkinter interface (human or AI play)
└── requirements.txt

````

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/alpha_go_lite.git
cd alpha_go_lite

# Python ≥3.9 and virtual-env recommended
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # torch, numpy, matplotlib, tqdm, etc.
````

*KataGo binaries* (for optional data generation) are already vendored under `katago_relatives/`, but you can drop in a newer network if you wish.

---

## 🚀 Quick Start

### 1. Train from scratch

```bash
python katago_relatives/katago-opencl-linux/networks/reinforcement_learning.py \
  --iterations 100 \        # RL cycles
  --games 50 \              # self-play games per cycle
  --epochs 25 \             # NN epochs per cycle
  --mcts_sims 800 \         # simulations per move
  --eval_simulations 3200 \ # simulations for evaluation matches
  --eval_games 20           # eval matches vs previous net
```

Checkpoints are written every cycle; resume with `--resume checkpoints/latest.pth`.

### 2. Play against the AI

```bash
python gui/play.py --size 9 --mcts_sims 800
```

| Option                         | Purpose             |
| ------------------------------ | ------------------- |
| `--size {5,9,13,19}`           | Board size          |
| `--human_first`                | Let human start     |
| `--model checkpoints/best.pth` | Load custom network |

---

## 📊 Training Curves & Metrics

* **KL-divergence** of the policy drops from 3.3 → 1.9 after 9 cycles, while **value MSE** reaches ≈ 0.23.
* **Entropy monitoring** stays near 2.2 nats, balancing exploration and exploit confidence.
* Win-rate progression shown below (20 games / point): rapid gains, then convergence around the 50 % line as nets level up .

*(Add `docs/plots/*.png` once your training run finishes for visual curves.)*

---

## 🔬 Research Internals

* **Minimal input planes** – only two binary channels (current / opponent stones) to force the net to infer tactics .
* **Top-k expansion** – keep k best policy moves during node expansion to cap tree width.
* **RAdam + weight-decay** with periodic *optimizer refresh* every 50 cycles to escape plateaus, following the ablation in the report .

---

## 🛣 Roadmap

* [ ] Add 13×13 profile & curriculum resize
* [ ] Graphical analyzer (heat-map of MCTS visits)
* [ ] Export to SGF & GTP bridge for online play
* [ ] Distributed self-play (Ray / MPI)

---

## 🤝 Contributing

1. Fork → branch (`feat/xxx`)
2. `pre-commit run --all-files` (black & flake8)
3. PR with a clear description, screenshots if GUI-related.

---

## 📜 License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## 🙏 Credits

* Inspired by DeepMind’s AlphaGo Zero and the **KataGo** open-source community.
* Built by the PE 50 team (École Centrale de Lyon, 2025) .

Happy Go-coding! 🏯♟️
