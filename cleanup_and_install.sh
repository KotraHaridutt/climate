#!/bin/bash

echo "=== Cleaning up to free disk space ==="

# Clear pip cache
echo "1. Clearing pip cache..."
pip cache purge

# Clean apt cache (if running as root or with sudo)
echo "2. Cleaning apt cache..."
apt-get clean 2>/dev/null || sudo apt-get clean 2>/dev/null || echo "  (skipped - no permissions)"

# Remove Python cache files
echo "3. Removing Python cache files..."
find /workspaces/climate -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find /workspaces/climate -type f -name "*.pyc" -delete 2>/dev/null

# Check disk space after cleanup
echo ""
echo "=== Disk space after cleanup ==="
df -h /

echo ""
echo "=== Installing lightweight PyTorch (CPU-only) and dependencies ==="

# Uninstall existing torch if present
pip uninstall -y torch torchvision torchaudio 2>/dev/null

# Install CPU-only PyTorch first (much smaller - ~200MB vs ~6GB)
echo "Installing PyTorch CPU-only version..."
pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Install other dependencies
echo "Installing other dependencies..."
pip install cdsapi xarray dask netCDF4 numpy streamlit requests folium streamlit-folium plotly fastapi uvicorn --no-cache-dir

echo ""
echo "=== Installation complete ==="
echo "Final disk space:"
df -h /
