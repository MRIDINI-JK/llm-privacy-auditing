#!/bin/bash

# LLM Privacy Audit Tool - Quick Setup Script

set -e

echo "======================================================================"
echo "  LLM PRIVACY AUDIT TOOL - SETUP"
echo "======================================================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
if (( $(echo "$python_version < 3.8" | bc -l) )); then
    echo "❌ Python 3.8+ required (found: $python_version)"
    exit 1
fi
echo "✓ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p outputs/{phase1,phase2,phase3,checkpoint_audits,api_jobs}
echo "✓ Directories created"

# Run quick test
echo ""
echo "======================================================================"
echo "  RUNNING QUICK TEST"
echo "======================================================================"
echo ""
python run_audit.py \
  --model gpt2 \
  --max_samples 5 \
  --output_dir outputs/quick_test \
  --device cpu

echo ""
echo "======================================================================"
echo "  SETUP COMPLETE!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Run full audit:"
echo "   python run_audit.py --model gpt2 --max_samples 50"
echo ""
echo "2. Start API server:"
echo "   python src/api.py"
echo "   Docs: http://localhost:8000/docs"
echo ""
echo "3. Start dashboard:"
echo "   python src/dashboard.py"
echo "   UI: http://localhost:8050"
echo ""
echo "4. Use Docker (recommended):"
echo "   docker-compose up -d"
echo ""
echo "See README.md for complete documentation."
echo ""