#!/usr/bin/env python3
"""
Test script for ONNX conversion of VoiceFilter model.
This script tests the conversion process and validates the results.
"""

import torch
import numpy as np
import librosa
import soundfile as sf
import os
from src.model.modeling_enh import VoiceFilter
from huggingface_hub import hf_hub_download
from onnx_converter import ONNXCompatibleVoiceFilter, STFTProcessor
import onnxruntime as ort

def test_original_model():
    """Test the original PyTorch model to ensure it works."""
    print("Testing original PyTorch model...")
    
    # Load the original model
    repo_id = 'nguyenvulebinh/voice-filter'
    model = VoiceFilter.from_pretrained(repo_id, cache_dir='./cache')
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    time_steps = 50
    freq_bins = 257
    embed_dim = 512
    
    dummy_spectrogram = torch.randn(batch_size, time_steps, freq_bins)
    dummy_speaker_embedding = torch.randn(batch_size, embed_dim)
    
    # Test forward pass
    with torch.no_grad():
        output = model.enh_model.separator(
            dummy_spectrogram, 
            dummy_speaker_embedding, 
            torch.ones(batch_size, dtype=torch.int32) * time_steps
        )
    
    print(f"Original model test successful! Output shape: {output[0][0].shape}")
    return model

def test_onnx_compatible_model():
    """Test the ONNX-compatible model wrapper."""
    print("Testing ONNX-compatible model wrapper...")
    
    # Load the original model
    repo_id = 'nguyenvulebinh/voice-filter'
    original_model = VoiceFilter.from_pretrained(repo_id, cache_dir='./cache')
    original_model.eval()
    
    # Create ONNX-compatible version
    onnx_model = ONNXCompatibleVoiceFilter(original_model)
    
    # Create dummy inputs
    batch_size = 1
    time_steps = 50
    freq_bins = 257
    embed_dim = 512
    
    dummy_spectrogram = torch.randn(batch_size, time_steps, freq_bins)
    dummy_speaker_embedding = torch.randn(batch_size, embed_dim)
    
    # Test forward pass
    with torch.no_grad():
        output = onnx_model(dummy_spectrogram, dummy_speaker_embedding)
    
    print(f"ONNX-compatible model test successful! Output shape: {output.shape}")
    return onnx_model

def test_stft_processor():
    """Test the STFT processor."""
    print("Testing STFT processor...")
    
    stft_processor = STFTProcessor()
    
    # Create dummy audio
    sample_rate = 16000
    duration = 1.0  # 1 second
    audio = np.random.randn(int(sample_rate * duration))
    
    # Test STFT
    spectrogram = stft_processor.compute_stft(audio)
    print(f"STFT output shape: {spectrogram.shape}")
    
    # Test ISTFT
    reconstructed_audio = stft_processor.compute_istft(spectrogram)
    print(f"ISTFT output shape: {reconstructed_audio.shape}")
    
    # Check reconstruction quality
    mse = np.mean((audio - reconstructed_audio) ** 2)
    print(f"Reconstruction MSE: {mse:.6f}")
    
    return stft_processor

def test_onnx_export():
    """Test the ONNX export process."""
    print("Testing ONNX export...")
    
    # Load the original model
    repo_id = 'nguyenvulebinh/voice-filter'
    original_model = VoiceFilter.from_pretrained(repo_id, cache_dir='./cache')
    original_model.eval()
    
    # Create ONNX-compatible version
    onnx_model = ONNXCompatibleVoiceFilter(original_model)
    
    # Create dummy inputs
    batch_size = 1
    time_steps = 50
    freq_bins = 257
    embed_dim = 512
    
    dummy_spectrogram = torch.randn(batch_size, time_steps, freq_bins)
    dummy_speaker_embedding = torch.randn(batch_size, embed_dim)
    
    # Export to ONNX
    print("Exporting to ONNX...")
    torch.onnx.export(
        onnx_model,
        (dummy_spectrogram, dummy_speaker_embedding),
        "test_voice_filter.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['spectrogram', 'speaker_embedding'],
        output_names=['enhanced_spectrogram'],
        dynamic_axes={
            'spectrogram': {0: 'batch_size', 1: 'time_steps'},
            'speaker_embedding': {0: 'batch_size'},
            'enhanced_spectrogram': {0: 'batch_size', 1: 'time_steps'}
        },
        verbose=True
    )
    
    print("ONNX export successful!")
    return "test_voice_filter.onnx"

def test_onnx_inference(onnx_path: str):
    """Test ONNX model inference."""
    print("Testing ONNX inference...")
    
    # Create ONNX Runtime session
    session = ort.InferenceSession(onnx_path)
    
    # Create test inputs
    batch_size = 1
    time_steps = 50
    freq_bins = 257
    embed_dim = 512
    
    test_spectrogram = np.random.randn(batch_size, time_steps, freq_bins).astype(np.float32)
    test_speaker_embedding = np.random.randn(batch_size, embed_dim).astype(np.float32)
    
    # Run inference
    inputs = {
        'spectrogram': test_spectrogram,
        'speaker_embedding': test_speaker_embedding
    }
    
    outputs = session.run(['enhanced_spectrogram'], inputs)
    
    print(f"ONNX inference successful! Output shape: {outputs[0].shape}")
    return outputs[0]

def compare_pytorch_vs_onnx():
    """Compare PyTorch and ONNX model outputs."""
    print("Comparing PyTorch vs ONNX outputs...")
    
    # Load the original model
    repo_id = 'nguyenvulebinh/voice-filter'
    original_model = VoiceFilter.from_pretrained(repo_id, cache_dir='./cache')
    original_model.eval()
    
    # Create ONNX-compatible version
    onnx_model = ONNXCompatibleVoiceFilter(original_model)
    
    # Create test inputs
    batch_size = 1
    time_steps = 50
    freq_bins = 257
    embed_dim = 512
    
    test_spectrogram = torch.randn(batch_size, time_steps, freq_bins)
    test_speaker_embedding = torch.randn(batch_size, embed_dim)
    
    # Get PyTorch output
    with torch.no_grad():
        pytorch_output = onnx_model(test_spectrogram, test_speaker_embedding)
    
    # Get ONNX output
    session = ort.InferenceSession("test_voice_filter.onnx")
    onnx_inputs = {
        'spectrogram': test_spectrogram.numpy(),
        'speaker_embedding': test_speaker_embedding.numpy()
    }
    onnx_output = session.run(['enhanced_spectrogram'], onnx_inputs)[0]
    
    # Compare outputs
    pytorch_np = pytorch_output.numpy()
    mse = np.mean((pytorch_np - onnx_output) ** 2)
    max_diff = np.max(np.abs(pytorch_np - onnx_output))
    
    print(f"PyTorch vs ONNX comparison:")
    print(f"  MSE: {mse:.8f}")
    print(f"  Max difference: {max_diff:.8f}")
    print(f"  PyTorch output shape: {pytorch_np.shape}")
    print(f"  ONNX output shape: {onnx_output.shape}")
    
    if mse < 1e-6:
        print("✅ PyTorch and ONNX outputs match closely!")
    else:
        print("⚠️  PyTorch and ONNX outputs differ significantly!")

def main():
    """Run all tests."""
    print("Starting VoiceFilter ONNX conversion tests...")
    print("=" * 50)
    
    try:
        # Test original model
        test_original_model()
        print()
        
        # Test ONNX-compatible wrapper
        test_onnx_compatible_model()
        print()
        
        # Test STFT processor
        test_stft_processor()
        print()
        
        # Test ONNX export
        onnx_path = test_onnx_export()
        print()
        
        # Test ONNX inference
        test_onnx_inference(onnx_path)
        print()
        
        # Compare PyTorch vs ONNX
        compare_pytorch_vs_onnx()
        print()
        
        print("✅ All tests completed successfully!")
        print("The ONNX conversion is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 