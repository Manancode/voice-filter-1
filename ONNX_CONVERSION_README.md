# VoiceFilter ONNX Conversion Guide

This guide explains how to convert the ConVoiFilter PyTorch model to ONNX format for Android deployment.

## Overview

The ConVoiFilter model is a Conformer-based voice filter that separates target speaker voices from mixed audio. Converting it to ONNX enables faster inference on mobile devices.

## Architecture

- **Model**: ConVoiFilter (Conformer-based voice filter)
- **Paper**: [ConVoiFilter: A Real-Time Neural Voice Filter](https://arxiv.org/pdf/2308.11380.pdf)
- **HuggingFace**: [nguyenvulebinh/voice-filter](https://huggingface.co/nguyenvulebinh/voice-filter)
- **Key Components**:
  - STFT encoder (512 FFT, 128 hop length)
  - Conformer separator (4 layers, 1024 attention dim, 8 heads)
  - XVector speaker embedding (512 dim)
  - PSM (Phase Sensitive Mask) for speech enhancement

## Conversion Strategy

Based on research findings, we use the following approach:

1. **Separate STFT operations** from the neural network
2. **Use ONNX opset version 17+** for better compatibility
3. **Handle chunking logic** in preprocessing
4. **Export speaker embedding model separately**

## Files

- `onnx_converter.py` - Basic ONNX conversion script
- `advanced_onnx_converter.py` - Advanced conversion with separate XVector model
- `test_conversion.py` - Test script to validate conversion
- `requirements_onnx.txt` - Dependencies for ONNX conversion

## Installation

```bash
# Install ONNX conversion dependencies
pip install -r requirements_onnx.txt

# Install original model dependencies
pip install -r requirements.txt
```

## Basic Conversion

### Step 1: Test the Conversion

```bash
python test_conversion.py
```

This will:
- Test the original PyTorch model
- Test the ONNX-compatible wrapper
- Test STFT processor
- Export to ONNX
- Test ONNX inference
- Compare PyTorch vs ONNX outputs

### Step 2: Convert to ONNX

```python
from onnx_converter import convert_to_onnx

# Convert the model
onnx_path = convert_to_onnx()
print(f"Model saved as: {onnx_path}")
```

## Advanced Conversion

For better performance, use the advanced converter that exports both models separately:

```python
from advanced_onnx_converter import (
    convert_voice_filter_to_onnx,
    convert_xvector_to_onnx,
    optimize_onnx_models
)

# Convert main model
main_onnx_path = convert_voice_filter_to_onnx()

# Convert XVector model
xvector_onnx_path = convert_xvector_to_onnx()

# Optimize for mobile
optimized_main, optimized_xvector = optimize_onnx_models()
```

## Usage in Android

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

### 3. Postprocessing

```python
def postprocess(enhanced_spectrogram):
    # Convert back to audio using ISTFT
    audio = stft_processor.compute_istft(enhanced_spectrogram)
    return audio
```

## Performance Optimization

### 1. Quantization

The advanced converter includes automatic quantization:

```python
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    model_input="voice_filter_main.onnx",
    model_output="voice_filter_main_quantized.onnx",
    weight_type=ort.quantization.QuantType.QUInt8
)
```

### 2. Mobile Optimization

For Android deployment, consider:

- **ONNX Runtime Mobile**: Optimized for mobile devices
- **TensorRT**: NVIDIA's inference engine (if using GPU)
- **Operator Fusion**: Combine multiple operations for efficiency

## Expected Performance

Based on research benchmarks:
- **RTF (Real-time Factor)**: 0.03-0.08 for Conformer models on mobile CPUs
- **Memory usage**: 64MB for quantized models
- **Latency**: <100ms for 5-second audio chunks

## Troubleshooting

### Common Issues

1. **STFT Export Error**
   - Solution: Use opset version 17+ and separate STFT operations

2. **Dynamic Shape Issues**
   - Solution: Configure dynamic axes properly in export

3. **Memory Layout Problems**
   - Solution: Ensure tensors are contiguous before export

4. **Conformer Attention Issues**
   - Solution: Use standard attention mechanisms supported by ONNX

### Error Messages

```
Exporting the operator stft to ONNX opset version 9 is not supported
```
→ Use opset version 17 or higher

```
ONNX export failed: Could not export operator
```
→ Check for unsupported operations in the model

## Research References

- [ESPnet-ONNX: Bridging a Gap Between Research and Production](https://arxiv.org/pdf/2209.09756.pdf)
- [ONNX Runtime Mobile Optimization](https://onnxruntime.ai/blogs/nimbleedge-x-onnxruntime)
- [PyTorch ONNX Export Tutorial](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html)

## License

This conversion code follows the same Apache 2.0 license as the original ConVoiFilter model. 