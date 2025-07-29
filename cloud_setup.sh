#!/bin/bash

echo "☁️  Cloud VM Setup for VoiceFilter ONNX Conversion"
echo "=================================================="

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python and pip
echo "🐍 Installing Python and pip..."
sudo apt install -y python3 python3-pip python3-venv

# Install system dependencies for audio processing
echo "🔊 Installing audio processing dependencies..."
sudo apt install -y libsndfile1-dev libasound2-dev portaudio19-dev

# Create virtual environment
echo "🔧 Creating Python virtual environment..."
python3 -m venv voice_filter_env
source voice_filter_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version for conversion)
echo "🔥 Installing PyTorch (CPU version)..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "📚 Installing other dependencies..."
pip install -r requirements_onnx.txt

# Clone the repository (if not already present)
if [ ! -d "voice-filter" ]; then
    echo "📥 Cloning voice-filter repository..."
    git clone https://github.com/nguyenvulebinh/voice-filter.git
    cd voice-filter
else
    echo "📁 Repository already exists"
    cd voice-filter
fi

# Create cache directory
mkdir -p cache

echo ""
echo "✅ Cloud VM setup completed!"
echo "==========================="
echo ""
echo "🚀 To start the conversion:"
echo "  1. Activate virtual environment: source voice_filter_env/bin/activate"
echo "  2. Run conversion: ./run_conversion.sh"
echo ""
echo "💡 Tips:"
echo "  - Use 'nohup ./run_conversion.sh &' to run in background"
echo "  - Monitor progress with 'tail -f nohup.out'"
echo "  - Download results with scp or cloud storage"
echo ""
echo "📊 Expected conversion time: 10-30 minutes"
echo "💾 Expected output size: 200-500MB" 