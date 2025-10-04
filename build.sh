#!/bin/bash

# Render Build Script
echo "🚀 Starting Render build process..."

# Update pip and install build tools
pip install --upgrade pip setuptools wheel

# Install requirements
echo "📦 Installing Python packages..."
pip install -r requirements.txt

echo "✅ Build completed successfully!"