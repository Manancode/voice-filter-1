#!/bin/bash

echo "🚀 Starting VoiceFilter ONNX Conversion Process"
echo "================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "📦 Installing ONNX conversion dependencies..."
pip3 install -r requirements_onnx.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies. Please check your Python environment."
    exit 1
fi

echo "✅ Dependencies installed successfully!"

echo "🧪 Running conversion tests..."
python3 test_conversion.py

if [ $? -ne 0 ]; then
    echo "❌ Conversion tests failed. Please check the error messages above."
    exit 1
fi

echo "✅ Conversion tests passed!"

echo "🔄 Running advanced conversion..."
python3 -c "
from advanced_onnx_converter import (
    convert_voice_filter_to_onnx,
    convert_xvector_to_onnx,
    optimize_onnx_models
)

print('Converting main model...')
main_onnx_path = convert_voice_filter_to_onnx()

print('Converting XVector model...')
xvector_onnx_path = convert_xvector_to_onnx()

print('Optimizing models for mobile...')
optimized_main, optimized_xvector = optimize_onnx_models()

print(f'✅ Conversion completed!')
print(f'Main model: {main_onnx_path}')
print(f'XVector model: {xvector_onnx_path}')
print(f'Optimized main model: {optimized_main}')
print(f'Optimized XVector model: {optimized_xvector}')
"

if [ $? -ne 0 ]; then
    echo "❌ Advanced conversion failed. Please check the error messages above."
    exit 1
fi

echo ""
echo "🎉 ONNX Conversion Completed Successfully!"
echo "=========================================="
echo ""
echo "📁 Generated Files:"
echo "  - voice_filter_main.onnx (Main model)"
echo "  - voice_filter_xvector.onnx (Speaker embedding model)"
echo "  - voice_filter_main_quantized.onnx (Optimized main model)"
echo "  - voice_filter_xvector_quantized.onnx (Optimized XVector model)"
echo ""
echo "📱 Ready for Android deployment!"
echo ""
echo "💡 Next steps:"
echo "  1. Copy the .onnx files to your Android project"
echo "  2. Use ONNX Runtime Mobile for inference"
echo "  3. Implement the preprocessing pipeline (STFT)"
echo "  4. Implement the postprocessing pipeline (ISTFT)"
echo ""
echo "📖 See ONNX_CONVERSION_README.md for detailed usage instructions." 