# LOINC AdaRank Ranking

This is a simple implementation of the AdaRank algorithm to rank relevant documents for a given query. The dataset used is the LOINC dataset, which is a set of lab tests and their corresponding codes.

## Setup

1. Clone the repository
```bash
git clone --recursive git@github.com:ikajdan/loinc_adarank_ranking.git
```

2. Set up the virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install the dependencies
```bash
pip install -r requirements.txt
```

4. Run the script
```bash
python main.py
```
