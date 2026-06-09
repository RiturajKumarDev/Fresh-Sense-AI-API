#!/bin/bash
# vercel-build.sh

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p static

# Copy model files if they're in the root
if [ -f "vegetable_classifier.pkl" ]; then
    echo "✅ Model files found in root"
fi

echo "Build completed successfully"