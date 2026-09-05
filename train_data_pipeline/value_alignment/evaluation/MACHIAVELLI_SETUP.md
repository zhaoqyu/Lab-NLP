# MACHIAVELLI Benchmark Setup Notes

MACHIAVELLI is an RL-style gym environment for text agents based on human-written Choose-Your-Own-Adventure games. It measures how agents trade off reward against ethically relevant behaviors such as deception, harm, manipulation, stealing, and other violations.

Official repo: https://github.com/aypan17/machiavelli

## Install

Use Python 3.11 or lower, as noted by the official repository.

```bash
git clone https://github.com/aypan17/machiavelli.git third_party/machiavelli
cd third_party/machiavelli
pip install -r requirements.txt
```

Download game data from the official Google Drive link in the MACHIAVELLI README and place it at:

```text
third_party/machiavelli/game_data/
```

The official README also describes an optional "artificial conscience" model under:

```text
third_party/machiavelli/model/
```

## Baseline Commands

Interactive or random-agent smoke test:

```bash
python -m demo -g <game_name> --agent RandomAgent
```

Generate trajectories:

```bash
python -m generate_trajectories \
  -a RandomAgent \
  --traj_dir ./demo_trajectories \
  --num_episodes 1
```

Evaluate trajectories:

```bash
python -m evaluate_trajectories \
  --traj_dir ./demo_trajectories \
  --results_file ./demo_results.csv
```

## How To Connect Our Trained Models

MACHIAVELLI custom agents inherit from `machiavelli.agent.BaseAgent` and implement `get_action`.

For our project, the custom agent should:

1. Load a Hugging Face base model and the target PEFT adapter.
2. Receive the current game observation and list of actions.
3. Use a neutral prompt that asks the model to choose one action index. Do not
   mention alignment or the target value in the evaluation prompt.
4. Return the selected action to MACHIAVELLI.

Keep this as an optional extrinsic benchmark because the full setup is heavier than AITA/KVS and requires separate game data downloads.

`machiavelli_hf_agent_template.py` uses the paper's stochastic decoding
settings (`temperature=0.6`, `top_p=0.9`, and `top_k=20`) and accepts an optional
PEFT adapter path. Keep the seed, game list, episode count, and decoding settings
identical for the base and conditioned model.

## Relevant Metrics For Our Report

- Game reward / achievement score.
- Ethical violations.
- Power-seeking behavior.
- Utility/disutility caused to characters.
- Trade-off between task reward and ethical behavior.

These metrics can be used as an additional behavioral transfer test after DPO/HyPO training.
