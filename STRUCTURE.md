Current structure
```
gpt/
├── data/
│   ├── __init__.py
│   ├── fineweb_data.py
|   ├── wikitext_data.py
│   ├── README.md
│   └── ...
├── models/
│   ├── __init__.py
│   ├── gpt2.py
│   └── ...
├── training/
│   ├── __init__.py
│   ├── train_gpt.py
│   ├── train_diffusion.py
│   └── ...
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

Proposed structure, including some testing scripts and some running scripts baked in, TODO later
```
gpt/
├── data/
│   ├── __init__.py
│   ├── fineweb_data.py
│   ├── README.md
│   └── ...
├── models/
│   ├── __init__.py
│   ├── gpt2.py
│   └── ...
├── training/
│   ├── __init__.py
│   ├── train_gpt.py
│   ├── train_diffusion.py
│   └── ...
├── scripts/
│   ├── download_data.py
│   ├── run_training.py
│   └── ...
├── tests/ (TBD)
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_training.py
│   └── ...
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

