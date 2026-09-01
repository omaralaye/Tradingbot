#!/bin/bash
# setup.sh — One-time project setup for Trading Bot
# Run this once: bash setup.sh

set -e
echo "=== Trading Bot Setup ==="

# 1. Install python3-venv if needed
if ! python3 -m venv --help &>/dev/null; then
    echo "Installing python3-venv..."
    sudo apt install -y python3.14-venv
fi

# 2. Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# 3. Activate and install dependencies
echo "Installing dependencies..."
source .venv/bin/activate
pip install --upgrade pip

# Core dependencies (without MetaTrader5 on Linux — handled separately)
pip install \
    pandas numpy ta \
    pydantic pydantic-settings python-dotenv \
    xgboost lightgbm shap scikit-learn \
    pyarrow loguru pyyaml joblib

# Dev dependencies
pip install pytest pytest-cov pytest-mock

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and fill in your MT5 credentials"
echo "  2. Run tests:  source .venv/bin/activate && pytest"
echo "  3. Run bot:    source .venv/bin/activate && python -m live.run_bot"
