# MuZero-style Agent for BitFall (JAX)

A compact, educational implementation of MuZero-inspired reinforcement learning in **JAX**, developed as coursework for **IT3105 AI Programming** at the Norwegian University of Science and Technology (NTNU).

**Developed by Lorenzo Cozzolino.**

The agent learns to play **BitFall**, a simple grid game in which falling debris interacts with a movable receptor. During training, tree search plans in a learned latent space; at evaluation time, a trained actor selects actions without search. This is **not** DeepMind's implementation and does not reproduce DeepMind's Atari, Go, chess or shogi experiments.

## Architecture

The code contains three neural networks:

- **Representation (`NNr`)**: encodes a window of one-hot game observations as a 128-dimensional latent state.
- **Dynamics (`NNd`)**: predicts a successor latent state and scalar immediate reward for a latent state and action.
- **Prediction (`NNp`)**: predicts action-policy logits and a scalar value from a latent state.

`mcts.py` uses the learned dynamics and prediction networks for **u-MCTS in latent space**, with visit counts and value estimates guiding search. `buffer.py` stores episodes (states, actions, rewards, search policies and values). `rl_manager.py` samples unrolled sequences and updates network weights using JAX automatic differentiation, a policy/value/reward objective and SGD. In **actor-only** mode, `NNr + NNp` choose a greedy action; **MCTS and `NNd` are not used at inference time**.

The environment in `simulator.py` is this project's Python implementation of **BitFall**. The rules and project motivation originate in the [official NTNU assignment](https://www.idi.ntnu.no/emner/it3105/assignments/muzero-knockoff.pdf); this repository does not redistribute the assignment document or lecture slides.

## Install

The source project recorded Python **3.14.2** and CPU execution, with packages pinned in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows activate with `.venv\Scripts\activate`. A compatible Python/JAX installation is needed; not all platforms are necessarily supported by these exact pinned versions. No API credentials or external data downloads are needed for the included environment, smoke checks or local evaluation.

## Check and run

Run the built-in simulator, target-generation and neural-network shape checks:

```bash
python -c "from main import run_verification_checks; run_verification_checks()"
```

Run the archived trained actor for 10 episodes:

```bash
python main.py --mode eval --weights run_20260424_1114_to_1424/muzero_weights_optimized.pkl --episodes 10
```

Run a **new** 100-episode actor-only benchmark, storing output separately from the archived results:

```bash
python benchmark_eval.py --weights run_20260424_1114_to_1424/muzero_weights_optimized.pkl --episodes 100 --output actor_eval_local.json
```

Create a gameplay GIF from the trained actor:

```bash
python make_bitfall_gif.py --weights run_20260424_1114_to_1424/muzero_weights_optimized.pkl --mode eval --output actor_demo.gif --fps 6 --max-steps 100 --seed 20
```

Training from scratch is available, but **not needed to use the archived weights**:

```bash
python main.py --mode train
```

The default configuration specifies 1,500 episodes, 50 search simulations per move, a maximum search depth of 6, a 6-step unroll, discount 0.95, and 100 steps per episode. A full training run can be time-consuming. See `config.py` for details. New runs write their own checkpoint and plots.

**Checkpoint security:** `muzero_weights_optimized.pkl` is serialized with Python pickle. Only unpickle a file you trust: opening an arbitrary malicious checkpoint can execute code. The archived file was inspected statically; such inspection cannot prove that arbitrary pickle input is safe.

## Archived evaluation results

The reference training run is `run_20260424_1114_to_1424/`. The results below are copied from **the original saved benchmark JSON files**; they are not claimed to be new measurements. Tests evaluate the **actor-only greedy policy**, with no search at inference time, in the included BitFall simulator, with `GLOBAL_SEED = 42`, a 100-step episode limit, and the recorded configuration.

| Evaluation | Episodes | Mean reward | Std. dev. | Min | Max | Recorded source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Primary | 1,000 | 6.902 | 5.634 | -14 | 23 | `run_20260424_1114_to_1424/benchmark_muzero_weights_optimized_1000eps.json` |
| Quick check | 100 | 7.990 | 5.516 | -3 | 22 | `run_20260424_1114_to_1424/benchmark_muzero_weights_optimized_100eps.json` |

The JSON retains per-episode rewards and configuration snapshots. Absolute personal paths in archived JSON metadata were replaced by paths relative to the repository; **metrics were not altered**. Newly generated benchmark and presentation summaries also use portable paths. Student-generated charts are in `results/presentation_figures/`; gameplay GIFs and the reference checkpoint are in the run folder.

The original full training-history log is **not included**. Therefore training trajectories cannot be reconstructed exactly from the published snapshot; rerunning the actor benchmark is possible with the saved weights. JAX versions and hardware may influence results.

## Files

| Files | Responsibility |
| --- | --- |
| `simulator.py`, `game_manager.py` | BitFall simulation, states and actions |
| `representations.py`, `neural_networks.py` | Observation encoding and JAX networks |
| `mcts.py` | Latent-space Monte Carlo tree search |
| `buffer.py`, `rl_manager.py` | Episode replay, training targets and gradient updates |
| `main.py`, `benchmark_eval.py` | Training, checks and evaluation |
| `make_bitfall_gif.py`, `generate_presentation_figures.py`, `visualization.py` | Game and result visualization |
| `config.py`, `requirements.txt` | Experiment settings and dependencies |
| `run_20260424_1114_to_1424/`, `results/` | Preserved reference weights, measurements and plots |

## Scope, attribution and rights

This is a **student-scale, BitFall-specific implementation**, not a general-purpose production RL library, a reproduction of DeepMind's benchmark scores, or a distribution of DeepMind's code.

The MuZero research concept is credited to **Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, Timothy Lillicrap and David Silver**: *Mastering Atari, Go, chess and shogi by planning with a learned model*, Nature 588, 604–609 (2020), [doi:10.1038/s41586-020-03051-4](https://doi.org/10.1038/s41586-020-03051-4).

JAX is an installed third-party library (Apache-2.0), not vendored in the repository. NTNU supplied the academic task and BitFall description; university PDFs, paper PDFs, and reconstructed lecture material are **not included**. See [source and rights references](docs/SOURCES_AND_RIGHTS.md).

**No open-source license has been assigned.** Redistribution of the code is subject to resolving author/collaborator rights and any applicable course sharing restrictions. Public visibility alone would not grant reuse permission.
