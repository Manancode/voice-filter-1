# URGENT: Find Exact ESPnet + Transformers Version Compatibility for ConVoiFilter ONNX Conversion

## Context
I'm converting a PyTorch Conformer voice filter model (ConVoiFilter from HuggingFace: nguyenvulebinh/voice-filter) to ONNX format. The model uses ESPnet framework and HuggingFace Transformers.

## Current Problem Chain
We've been through multiple import errors, each time fixing one but hitting another:

### Error 1: ✅ FIXED - NumPy Inf Issue
```
ImportError: cannot import name 'Inf' from 'numpy'
```
**SOLUTION**: Used NumPy 1.24.3, SciPy 1.10.1 (confirmed by research)

### Error 2: ❌ CURRENT - Transformers Compatibility
```
ImportError: cannot import name 'get_cached_models' from 'transformers.utils'
```
**ATTEMPTED**: Transformers 4.31.0 → 4.26.0 → 4.25.0

### Error 3: ❌ CURRENT - Transformers pytorch_utils Issue
```
ImportError: cannot import name 'torch_int_div' from 'transformers.pytorch_utils'
```

## Specific Questions to Research

### 1. ESPnet Version Compatibility
- **What Transformers version does ESPnet 202308 officially support?**
- **What Transformers version does ESPnet 202304 support?**
- **What's the oldest ESPnet version that works with the ConVoiFilter model?**
- **Are there known compatibility issues between ESPnet and specific Transformers versions?**

### 2. Transformers Version Timeline
- **When was `get_cached_models` removed from transformers.utils?**
- **When was `torch_int_div` removed from transformers.pytorch_utils?**
- **What Transformers version still has both these functions?**
- **What's the exact Transformers version where these breaking changes occurred?**

### 3. ConVoiFilter Model Requirements
- **What Transformers version was used to train the ConVoiFilter model?**
- **What ESPnet version was used to train the ConVoiFilter model?**
- **Are there specific version requirements mentioned in the model's config or documentation?**

### 4. Working Version Combinations
- **What's the EXACT working combination of:**
  - ESPnet version
  - Transformers version
  - NumPy version (1.24.3)
  - SciPy version (1.10.1)
  - PyTorch version (1.13.1)
  - Python version (3.10)

### 5. Alternative Approaches
- **Can we use an older ESPnet version that's compatible with newer Transformers?**
- **Can we patch the ESPnet code to work with newer Transformers?**
- **Are there forks or modified versions of ESPnet that work with newer Transformers?**
- **Can we use a different approach to load the model without the full ESPnet framework?**

### 6. Model Loading Alternatives
- **Can we load the ConVoiFilter model without using the ESPnet framework?**
- **Is there a way to extract just the PyTorch model weights and load them directly?**
- **Can we use HuggingFace's model loading without the ESPnet dependencies?**

## Research Sources to Check
1. **ESPnet GitHub repository** - Check requirements.txt, setup.py, and documentation
2. **ESPnet release notes** - Version compatibility information
3. **Transformers GitHub repository** - Breaking changes and version history
4. **Transformers release notes** - When functions were removed
5. **ConVoiFilter model repository** - Check model requirements and dependencies
6. **Stack Overflow** - Search for "ESPnet transformers compatibility" or "get_cached_models torch_int_div"
7. **GitHub issues** - Search ESPnet and Transformers repositories for compatibility issues
8. **ESPnet documentation** - Official compatibility matrix
9. **HuggingFace model cards** - Check if ConVoiFilter has specific version requirements

## Current Environment
- Python 3.10
- NumPy 1.24.3 ✅
- SciPy 1.10.1 ✅
- PyTorch 1.13.1 ✅
- ESPnet 202308 ❌ (trying different versions)
- Transformers 4.25.0 ❌ (trying different versions)

## Expected Outcome
I need the exact version combination that works together, or an alternative approach to load the ConVoiFilter model without these compatibility issues.

## Additional Context
- The goal is ONNX conversion for Android deployment
- The model is 2 years old, so it was built with older library versions
- We need to maintain compatibility with the model's expected environment
- The user wants ONLY ONNX files as output, no PyTorch dependencies in final Android app

## Urgency
This is blocking the entire ONNX conversion process. We need a definitive solution to proceed with the model conversion. 