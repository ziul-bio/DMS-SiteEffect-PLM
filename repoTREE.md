DMS-SiteEffect-PLM/
├── data/                   # Raw and processed datasets, to keep all the 51 DMS
│   ├── raw/                ## Data used on domain adaptation
│   └── processed/
│   └── cellular/ and viral/ ## DMS datasets
├── scripts/                # scripts such as: finetune, lasso, extract embeddings, OLS etc.
├── src/
|   ├── models/             # LLM model definitions & wrappers
|   │   ├── model_MLM.py       # Load base model architecture.
|   |   ├── esm2_config.py     # model architecture parameters
|   │
|   ├── data/               # Data loading & tokenization
|   │   ├── dataset.py         # Custom PyTorch Dataset classes
|   │   └── preprocess.py      # Data cleaning, formatting, prompt building
|   │
|   ├── training/           # Training logic
|   │   ├── train_esm2.py           # Trainer class from pytorch lighting for the last 3 layers
|   │   ├── train_esm2_partial.py    # Trainer class from pytorch lighting for the full training
|   │
├── notebooks/              # Exploratory notebooks, and results figures
├── experiments/            # Logs, model outputs, results
├── checkpoints/            # Saved models
├── logs/                   # Training and evaluation logs
├── requirements.txt        # Python dependencies
├── README.md
└── .gitignore
