#!/bin/bash
# setup_machiavelli.sh
# Script to setup MACHIAVELLI benchmark for evaluation

echo "Setting up MACHIAVELLI Benchmark..."

# Clone the repository
if [ ! -d "machiavelli" ]; then
    git clone https://github.com/aypan17/machiavelli.git
fi

cd machiavelli || exit

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo "MACHIAVELLI setup complete. Ready to evaluate Power-Seeking, Moral Violations, Disutility, and Game Score."
