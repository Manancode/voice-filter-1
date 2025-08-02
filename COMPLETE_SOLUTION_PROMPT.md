# URGENT: Complete Environment Setup for ConVoiFilter ONNX Conversion - ALL ISSUES

## Context
Converting PyTorch Conformer voice filter model (ConVoiFilter from HuggingFace: nguyenvulebinh/voice-filter) to ONNX format. The model uses ESPnet framework and HuggingFace Transformers.

## Current Repository Structure
- Original `requirements.txt` (from repo): transformers==4.24.0, espnet==202209, etc.
- Modified `requirements_onnx.txt` (we created): torch>=1.13.0, torchaudio>=0.13.0, etc.
- Both files are conflicting and causing version issues

## ALL Issues We've Faced

### 1. NumPy Inf Issue ✅ FIXED
```
ImportError: cannot import name 'Inf' from 'numpy'
```
**SOLUTION**: NumPy 1.24.3, SciPy 1.10.1

### 2. Transformers Compatibility Issues ✅ FIXED
```
ImportError: cannot import name 'get_cached_models' from 'transformers.utils'
ImportError: cannot import name 'torch_int_div' from 'transformers.pytorch_utils'
```
**SOLUTION**: Transformers 4.21.0

### 3. PyTorch/torchaudio Version Mismatch ❌ CURRENT
```
undefined symbol: _ZNK5torch8autograd4Node4nameEv
cannot import name 'fail_with_message' from 'torchaudio._internal.module_utils'
```
**PROBLEM**: Mixed conda/pip installations, version conflicts

### 4. ESPnet Compatibility ❌ CURRENT
- ESPnet 202209 (from original requirements.txt)
- ESPnet 202304 (we tried)
- Version conflicts with Transformers

## Specific Questions for Research

### 1. Exact Working Environment
**What's the EXACT working combination of ALL packages:**
- Python version
- PyTorch version
- torchaudio version
- ESPnet version
- Transformers version
- NumPy version
- SciPy version
- All other dependencies

### 2. Installation Method
- **Should we use conda or pip?**
- **Should we use PyTorch index or conda-forge?**
- **What's the exact installation order?**
- **How to avoid mixed conda/pip installations?**

### 3. Repository Requirements Analysis
- **Which requirements file should we use?** (requirements.txt vs requirements_onnx.txt)
- **What versions were used to train the original ConVoiFilter model?**
- **Are there specific version constraints in the model config?**

### 4. Environment Setup Strategy
- **Should we create a fresh conda environment?**
- **Should we use a specific Python version?**
- **How to ensure all packages are from the same source?**

### 5. Alternative Approaches
- **Can we use a different ESPnet version that's compatible with newer PyTorch?**
- **Can we modify the model code to work with newer versions?**
- **Is there a way to load the model without the full ESPnet framework?**

## Current Failed Attempts
1. PyTorch 1.13.1 + torchaudio 0.13.1 (symbol mismatch)
2. PyTorch 2.7.1 + torchaudio 2.7.1 (import error)
3. Mixed conda/pip installations (always fails)
4. Different ESPnet versions (compatibility issues)

## Expected Outcome
I need:
1. **Exact package versions** that work together
2. **Exact installation commands** in the right order
3. **Complete environment setup** that avoids all conflicts
4. **Verification steps** to confirm everything works

## Additional Context
- Goal: ONNX conversion for Android deployment
- Model is 2 years old, needs older library compatibility
- User wants ONLY ONNX files as output
- Current environment has mixed conda/pip installations causing symbol conflicts

## Urgency
This is blocking the entire ONNX conversion process. We need a definitive, tested solution that works from start to finish. 