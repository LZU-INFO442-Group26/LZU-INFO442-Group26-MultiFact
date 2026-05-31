#!/bin/bash
# =============================================================================
# XFacta Dataset Download Script
# =============================================================================
# Downloads the raw XFacta dataset from Google Drive.
#
# Prerequisites:
#   - gdown (pip install gdown)
#   - Or manually download from the Google Drive link below
#
# Usage:
#   chmod +x download_data.sh
#   ./download_data.sh
#
# Source: https://drive.google.com/drive/folders/1Sj5Rr6TpbPNzWhUjQt60fRc6xSQD2DWK
# Paper:  https://arxiv.org/abs/2508.09999
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR"

echo "=========================================="
echo " XFacta Dataset Download"
echo "=========================================="

# Option 1: Download using gdown (recommended if available)
if command -v gdown &> /dev/null; then
    echo ""
    echo "[1/2] Downloading dataset from Google Drive..."
    gdown --folder "https://drive.google.com/drive/folders/1Sj5Rr6TpbPNzWhUjQt60fRc6xSQD2DWK" \
        -O "$DATA_DIR" \
        --remaining-ok

    echo ""
    echo "[2/2] Download complete!"

# Option 2: Manual download instructions
else
    echo ""
    echo "  'gdown' not found. Install it with:  pip install gdown"
    echo ""
    echo "  Alternatively, download manually from:"
    echo "  https://drive.google.com/drive/folders/1Sj5Rr6TpbPNzWhUjQt60fRc6xSQD2DWK"
    echo ""
    echo "  Place the downloaded files in: $DATA_DIR"
    echo ""
    echo "  Expected structure after download:"
    echo "    $DATA_DIR/"
    echo "    ├── dev.json"
    echo "    ├── test.json"
    echo "    ├── real_sample/"
    echo "    │   └── batch*.json"
    echo "    ├── fake_sample/"
    echo "    │   └── batch*.json"
    echo "    └── {sample}_sample/media/"
    echo "        └── batch*/"
    echo "            └── */images/*.jpeg"
fi

echo ""
echo "=========================================="
echo " After download, run the preprocessing:"
echo "   python3 xfacta_data/preprocess.py"
echo "=========================================="
