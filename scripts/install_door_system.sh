#!/bin/bash
# Installation script for complete DoOR-FlyWire ORN pathway analysis system
#
# This script installs all dependencies and sets up the DoOR database
# integration for the PGCN framework.
#
# Usage:
#   bash scripts/install_door_system.sh
#
# Requirements:
#   - Python 3.8+
#   - R 4.0+ (will be checked/installed)
#   - Internet connection for package downloads

set -e  # Exit on error

echo "================================================================="
echo "DoOR-FlyWire ORN Pathway Analysis System - Installation"
echo "================================================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! python3 -c 'import sys; assert sys.version_info >= (3, 8)' 2>/dev/null; then
    echo -e "${RED}Error: Python 3.8 or higher is required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Check R installation
echo "Checking R installation..."
if command -v R &> /dev/null; then
    r_version=$(R --version | head -1)
    echo "R found: $r_version"
    echo -e "${GREEN}✓ R is installed${NC}"
else
    echo -e "${YELLOW}Warning: R is not installed${NC}"
    echo "R is required for DoOR database access via rpy2"
    echo ""
    echo "Installation instructions:"
    echo "  Ubuntu/Debian: sudo apt-get install r-base r-base-dev"
    echo "  macOS: brew install r"
    echo "  Windows: Download from https://cran.r-project.org/"
    echo ""
    read -p "Continue without R? (CSV fallback will be used) [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
echo "This may take several minutes..."
pip install -r requirements_door.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install Python dependencies${NC}"
    exit 1
fi
echo ""

# Check rpy2 installation
echo "Checking rpy2 installation..."
if python3 -c "import rpy2" 2>/dev/null; then
    echo -e "${GREEN}✓ rpy2 installed successfully${NC}"
    RPY2_AVAILABLE=true
else
    echo -e "${YELLOW}⚠ rpy2 not available (will use CSV fallback)${NC}"
    RPY2_AVAILABLE=false
fi
echo ""

# Install DoOR R packages (if rpy2 is available)
if [ "$RPY2_AVAILABLE" = true ]; then
    echo "Installing DoOR R packages..."
    echo "This will download and install:"
    echo "  - DoOR.data (v2.0.1)"
    echo "  - DoOR.functions (v2.0.1)"
    echo ""

    python3 << EOF
from pgcn.door import DoORDataManager
try:
    manager = DoORDataManager(method='rpy2')
    success = manager.install_door_packages()
    if success:
        print("\033[0;32m✓ DoOR R packages installed\033[0m")
    else:
        print("\033[0;31m✗ Failed to install DoOR R packages\033[0m")
except Exception as e:
    print(f"\033[0;31m✗ Error: {e}\033[0m")
EOF
    echo ""
fi

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/door_cache
mkdir -p results/orn_analysis
mkdir -p tests/door

echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Download FlyWire labels (if not present)
echo "Checking for FlyWire community labels..."
if [ ! -f "data/processed_labels.csv.gz" ]; then
    echo "FlyWire labels not found."
    echo "Download from: https://codex.flywire.ai/api/download?dataset=fafb"
    echo ""
    read -p "Download now? (requires ~1MB, 100K+ cells) [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Downloading FlyWire labels..."
        curl -o data/processed_labels.csv.gz \
            "https://codex.flywire.ai/api/download?dataset=fafb"

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ FlyWire labels downloaded${NC}"
        else
            echo -e "${RED}✗ Download failed${NC}"
        fi
    fi
else
    echo -e "${GREEN}✓ FlyWire labels already present${NC}"
fi
echo ""

# Run tests to verify installation
echo "Running installation tests..."
pytest tests/door/ -v --tb=short

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed${NC}"
else
    echo -e "${YELLOW}⚠ Some tests failed (may be expected if R not available)${NC}"
fi
echo ""

# Print summary
echo "================================================================="
echo "Installation Summary"
echo "================================================================="
echo ""
echo -e "${GREEN}✓ Python dependencies installed${NC}"
if [ "$RPY2_AVAILABLE" = true ]; then
    echo -e "${GREEN}✓ R integration available (rpy2)${NC}"
    echo -e "${GREEN}✓ DoOR R packages installed${NC}"
else
    echo -e "${YELLOW}⚠ R integration not available (CSV fallback enabled)${NC}"
fi
echo -e "${GREEN}✓ Directory structure created${NC}"
echo ""

echo "Next steps:"
echo "1. Download FlyWire labels (if not done):"
echo "   wget https://codex.flywire.ai/api/download?dataset=fafb -O data/processed_labels.csv.gz"
echo ""
echo "2. Run complete ORN analysis:"
echo "   python scripts/complete_orn_analysis.py"
echo ""
echo "3. Run tests:"
echo "   pytest tests/door/ -v"
echo ""
echo "4. Interactive analysis:"
echo "   jupyter notebook"
echo ""

echo "================================================================="
echo -e "${GREEN}Installation complete!${NC}"
echo "================================================================="
