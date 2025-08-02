# URGENT: Find Exact torchaudio Version for PyTorch 1.13.1 + ESPnet 202304

## Context
We're converting a ConVoiFilter model to ONNX and have fixed most compatibility issues, but now have a torchaudio error:

```
undefined symbol: _ZNK5torch8autograd4Node4nameEv
```

## Current Environment
- PyTorch: 1.13.1
- ESPnet: 202304
- Transformers: 4.21.0
- NumPy: 1.24.3
- SciPy: 1.10.1
- Python: 3.10

## Specific Questions

### 1. PyTorch 1.13.1 + torchaudio Compatibility
- **What torchaudio version is officially compatible with PyTorch 1.13.1?**
- **What's the exact torchaudio version that was released with PyTorch 1.13.1?**
- **Are there known compatibility issues between PyTorch 1.13.1 and specific torchaudio versions?**

### 2. ESPnet 202304 + torchaudio Requirements
- **What torchaudio version does ESPnet 202304 officially require?**
- **What torchaudio version was used when ESPnet 202304 was released?**
- **Are there specific torchaudio version constraints in ESPnet 202304?**

### 3. The Specific Error
- **What causes the `undefined symbol: _ZNK5torch8autograd4Node4nameEv` error?**
- **Is this a known issue with specific PyTorch/torchaudio version combinations?**
- **What's the exact fix for this symbol error?**

### 4. Alternative Solutions
- **Can we install torchaudio from the same source as PyTorch (PyTorch index)?**
- **Should we use `pip install torchaudio --index-url https://download.pytorch.org/whl/cpu`?**
- **Are there specific installation commands for matching versions?**

## Research Sources
1. **PyTorch release notes** - Check what torchaudio version was released with PyTorch 1.13.1
2. **ESPnet requirements** - Check ESPnet 202304 requirements.txt or setup.py
3. **PyTorch documentation** - Official compatibility matrix
4. **GitHub issues** - Search for "undefined symbol torchaudio" or "torchaudio compatibility"
5. **Stack Overflow** - Similar torchaudio version issues

## Expected Outcome
I need the exact torchaudio version that works with PyTorch 1.13.1 and ESPnet 202304, or the correct installation method to ensure compatibility. 