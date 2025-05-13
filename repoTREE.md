llm-project/
├── data/                   # Raw and processed datasets, to keep all the 51 DMS
│   ├── raw/
│   └── processed/
├── scripts/                # One-off data processing, training, or evaluation scripts, fine-tune scripts etc.
├── src/
|   ├── models/             # LLM model definitions & wrappers
|   │   ├── model.py           # Core model architecture (e.g., GPT, BERT)
|   │   ├── model_utils.py     # Utilities for loading/saving/checkpointing
|   |   ├── model_config.py    # model architecture parameters
|   │   └── adapters.py        # (Optional) LoRA/adapter modules
|   │
|   ├── data/               # Data loading & tokenization
|   │   ├── dataset.py         # Custom PyTorch Dataset classes
|   │   ├── tokenizer.py       # Tokenizer loading/wrapping
|   │   └── preprocess.py      # Data cleaning, formatting, prompt building
|   │
|   ├── training/           # Training logic
|   │   ├── train.py           # Training loop or Trainer class
|   │   ├── optimizer.py       # Optimizer & scheduler config
|   │   ├── loss.py            # Custom loss functions if any
|   │   └── callbacks.py       # Logging, early stopping, checkpointing
|   │
|   ├── evaluation/         # Evaluation, metrics, benchmarking
|   │   ├── eval.py            # Evaluation loop
|   │   ├── metrics.py         # Accuracy, perplexity, BLEU, etc.
|   │   └── inference.py       # Generate/predict from a trained model
|   │
|   ├── utils/              # General utilities
|   │   ├── logging.py         # Logging setup (e.g., wandb, tensorboard)
|   │   ├── config.py          # Config loader (e.g., OmegaConf, argparse)
|   │   └── misc.py            # Other helpers: seeding, device setup
|   │
|   └── main.py             # Entry point to training/evaluation
├── configs/                # YAML/JSON config files for runs
├── notebooks/              # Exploratory notebooks
├── experiments/            # Logs, model outputs, results
├── checkpoints/            # Saved models
├── logs/                   # Training and evaluation logs
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── README.md
└── .gitignore
