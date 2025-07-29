# VoiceFilter ONNX Converter

Convert the ConVoiFilter PyTorch model to ONNX format for Android deployment with optimized performance.

## 🎯 Overview

This repository provides tools to convert the [ConVoiFilter](https://huggingface.co/nguyenvulebinh/voice-filter) PyTorch model to ONNX format, enabling fast inference on mobile devices. The ConVoiFilter is a Conformer-based voice filter that separates target speaker voices from mixed audio.

## 🏗️ Architecture

- **Model**: ConVoiFilter (Conformer-based voice filter)
- **Paper**: [ConVoiFilter: A Real-Time Neural Voice Filter](https://arxiv.org/pdf/2308.11380.pdf)
- **HuggingFace**: [nguyenvulebinh/voice-filter](https://huggingface.co/nguyenvulebinh/voice-filter)
- **Key Components**:
  - STFT encoder (512 FFT, 128 hop length)
  - Conformer separator (4 layers, 1024 attention dim, 8 heads)
  - XVector speaker embedding (512 dim)
  - PSM (Phase Sensitive Mask) for speech enhancement

## 🚀 Quick Start

### Local Setup

```bash
# Clone this repository
git clone <your-repo-url>
cd voice-filter-onnx-converter

# Install dependencies
pip install -r requirements_onnx.txt

# Run conversion
./run_conversion.sh
```

### Cloud Setup (Recommended)

```bash
# On your cloud VM
chmod +x cloud_setup.sh
./cloud_setup.sh

# Activate environment and run conversion
source voice_filter_env/bin/activate
./run_conversion.sh
```

## 📁 Files

- `onnx_converter.py` - Basic ONNX conversion script
- `advanced_onnx_converter.py` - Advanced conversion with separate XVector model
- `test_conversion.py` - Test script to validate conversion
- `run_conversion.sh` - Automated conversion script
- `cloud_setup.sh` - Cloud VM setup script
- `requirements_onnx.txt` - Dependencies for ONNX conversion
- `ONNX_CONVERSION_README.md` - Detailed conversion guide

## 🔧 Conversion Strategy

Based on research findings, we use the following approach:

1. **Separate STFT operations** from the neural network
2. **Use ONNX opset version 17+** for better compatibility
3. **Handle chunking logic** in preprocessing
4. **Export speaker embedding model separately**
5. **Quantize models** for mobile optimization

## 📱 Android Integration

### 1. Preprocessing Pipeline

```python
class VoiceFilterPreprocessor:
    def __init__(self):
        self.stft_processor = STFTProcessor()
    
    def process_audio(self, mixed_audio, reference_audio):
        # 1. Compute STFT of mixed audio
        spectrogram = self.stft_processor.compute_stft(mixed_audio)
        
        # 2. Compute speaker embedding from reference
        speaker_embedding = self.compute_speaker_embedding(reference_audio)
        
        return spectrogram, speaker_embedding
```

### 2. ONNX Inference

```python
import onnxruntime as ort

class VoiceFilterONNX:
    def __init__(self, onnx_path):
        self.session = ort.InferenceSession(onnx_path)
    
    def enhance(self, spectrogram, speaker_embedding):
        inputs = {
            'spectrogram': spectrogram,
            'speaker_embedding': speaker_embedding
        }
        enhanced_spec = self.session.run(['enhanced_spectrogram'], inputs)[0]
        return enhanced_spec
```

## 📊 Performance

Based on research benchmarks:
- **RTF (Real-time Factor)**: 0.03-0.08 for Conformer models on mobile CPUs
- **Memory usage**: 64MB for quantized models
- **Latency**: <100ms for 5-second audio chunks

## 🛠️ Troubleshooting

### Common Issues

1. **STFT Export Error**
   - Solution: Use opset version 17+ and separate STFT operations

2. **Dynamic Shape Issues**
   - Solution: Configure dynamic axes properly in export

3. **Memory Layout Problems**
   - Solution: Ensure tensors are contiguous before export

4. **Conformer Attention Issues**
   - Solution: Use standard attention mechanisms supported by ONNX

## 📚 Research References

- [ESPnet-ONNX: Bridging a Gap Between Research and Production](https://arxiv.org/pdf/2209.09756.pdf)
- [ONNX Runtime Mobile Optimization](https://onnxruntime.ai/blogs/nimbleedge-x-onnxruntime)
- [PyTorch ONNX Export Tutorial](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original ConVoiFilter model by [nguyenvulebinh](https://huggingface.co/nguyenvulebinh)
- ESPnet community for the base architecture
- ONNX Runtime team for mobile optimization tools

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Made with ❤️ for the speech processing community**