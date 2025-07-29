import torch
import torch.onnx
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional
import os
from src.model.modeling_enh import VoiceFilter
from huggingface_hub import hf_hub_download

class ONNXCompatibleVoiceFilter(torch.nn.Module):
    """
    ONNX-compatible version of VoiceFilter model.
    Separates STFT operations from the neural network for ONNX export.
    """
    
    def __init__(self, original_model: VoiceFilter):
        super().__init__()
        self.enh_model = original_model.enh_model
        self.filter_condition_transform = original_model.filter_condition_transform
        self.dropout = original_model.dropout
        
        # Set models to eval mode for ONNX export
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, spectrogram: torch.Tensor, speaker_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ONNX export.
        
        Args:
            spectrogram: Pre-computed STFT features [batch_size, time, freq]
            speaker_embedding: Speaker embedding [batch_size, embed_dim]
            
        Returns:
            Enhanced spectrogram [batch_size, time, freq]
        """
        # Process through the enhancement model
        speech_lengths = torch.ones(spectrogram.size(0), dtype=torch.int32) * spectrogram.size(1)
        
        # Forward through the separator (Conformer)
        enhanced_spec, _, _ = self.enh_model.separator(
            spectrogram, 
            speaker_embedding, 
            speech_lengths
        )
        
        # Return the enhanced spectrogram (first speaker)
        return enhanced_spec[0]

class ONNXCompatibleXVector(torch.nn.Module):
    """
    ONNX-compatible version of XVector model for speaker embedding.
    """
    
    def __init__(self, original_model: VoiceFilter):
        super().__init__()
        self.xvector_model = original_model.xvector_model
        
        # Set model to eval mode for ONNX export
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for speaker embedding.
        
        Args:
            audio: Audio tensor [batch_size, 1, time]
            
        Returns:
            Speaker embedding [batch_size, embed_dim]
        """
        return self.xvector_model(audio)

class STFTProcessor:
    """
    Handles STFT/ISTFT operations outside of ONNX model.
    """
    
    def __init__(self, n_fft: int = 512, hop_length: int = 128):
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def compute_stft(self, audio: np.ndarray) -> np.ndarray:
        """Compute STFT features."""
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        # Convert to magnitude spectrogram
        magnitude = np.abs(stft)
        # Transpose to [time, freq] format
        return magnitude.T
    
    def compute_istft(self, spectrogram: np.ndarray) -> np.ndarray:
        """Compute ISTFT from spectrogram."""
        # Transpose back to [freq, time] format
        stft = spectrogram.T
        # Reconstruct audio (assuming phase from original or using Griffin-Lim)
        audio = librosa.istft(stft, hop_length=self.hop_length)
        return audio

def convert_voice_filter_to_onnx():
    """Convert the main VoiceFilter model to ONNX format."""
    
    print("Loading original model...")
    # Load the original model
    repo_id = 'nguyenvulebinh/voice-filter'
    original_model = VoiceFilter.from_pretrained(repo_id, cache_dir='./cache')
    original_model.eval()
    
    print("Creating ONNX-compatible VoiceFilter model...")
    # Create ONNX-compatible version
    onnx_model = ONNXCompatibleVoiceFilter(original_model)
    
    # Create dummy inputs for ONNX export
    batch_size = 1
    time_steps = 100  # Adjust based on your needs
    freq_bins = 257   # (n_fft//2 + 1)
    embed_dim = 512
    
    dummy_spectrogram = torch.randn(batch_size, time_steps, freq_bins)
    dummy_speaker_embedding = torch.randn(batch_size, embed_dim)
    
    print("Exporting VoiceFilter to ONNX...")
    # Export to ONNX with dynamic axes
    dynamic_axes = {
        'spectrogram': {0: 'batch_size', 1: 'time_steps'},
        'speaker_embedding': {0: 'batch_size'},
        'enhanced_spectrogram': {0: 'batch_size', 1: 'time_steps'}
    }
    
    torch.onnx.export(
        onnx_model,
        (dummy_spectrogram, dummy_speaker_embedding),
        "voice_filter_main.onnx",
        export_params=True,
        opset_version=17,  # Use latest opset for better STFT support
        do_constant_folding=True,
        input_names=['spectrogram', 'speaker_embedding'],
        output_names=['enhanced_spectrogram'],
        dynamic_axes=dynamic_axes,
        verbose=True
    )
    
    print("VoiceFilter ONNX model exported successfully!")
    return "voice_filter_main.onnx"

def convert_xvector_to_onnx():
    """Convert the XVector model to ONNX format."""
    
    print("Loading original model for XVector...")
    # Load the original model
    repo_id = 'nguyenvulebinh/voice-filter'
    original_model = VoiceFilter.from_pretrained(repo_id, cache_dir='./cache')
    original_model.eval()
    
    print("Creating ONNX-compatible XVector model...")
    # Create ONNX-compatible version
    onnx_xvector = ONNXCompatibleXVector(original_model)
    
    # Create dummy inputs for ONNX export
    batch_size = 1
    audio_length = 80000  # 5 seconds at 16kHz
    
    dummy_audio = torch.randn(batch_size, 1, audio_length)
    
    print("Exporting XVector to ONNX...")
    # Export to ONNX with dynamic axes
    dynamic_axes = {
        'audio': {0: 'batch_size', 2: 'audio_length'},
        'speaker_embedding': {0: 'batch_size'}
    }
    
    torch.onnx.export(
        onnx_xvector,
        dummy_audio,
        "voice_filter_xvector.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['audio'],
        output_names=['speaker_embedding'],
        dynamic_axes=dynamic_axes,
        verbose=True
    )
    
    print("XVector ONNX model exported successfully!")
    return "voice_filter_xvector.onnx"

def test_onnx_models():
    """Test both exported ONNX models."""
    import onnxruntime as ort
    
    print("Testing ONNX models...")
    
    # Test main model
    print("Testing main VoiceFilter model...")
    main_session = ort.InferenceSession("voice_filter_main.onnx")
    
    batch_size = 1
    time_steps = 50
    freq_bins = 257
    embed_dim = 512
    
    test_spectrogram = np.random.randn(batch_size, time_steps, freq_bins).astype(np.float32)
    test_speaker_embedding = np.random.randn(batch_size, embed_dim).astype(np.float32)
    
    main_inputs = {
        'spectrogram': test_spectrogram,
        'speaker_embedding': test_speaker_embedding
    }
    
    main_outputs = main_session.run(['enhanced_spectrogram'], main_inputs)
    print(f"Main model test successful! Output shape: {main_outputs[0].shape}")
    
    # Test XVector model
    print("Testing XVector model...")
    xvector_session = ort.InferenceSession("voice_filter_xvector.onnx")
    
    audio_length = 80000
    test_audio = np.random.randn(batch_size, 1, audio_length).astype(np.float32)
    
    xvector_inputs = {
        'audio': test_audio
    }
    
    xvector_outputs = xvector_session.run(['speaker_embedding'], xvector_inputs)
    print(f"XVector model test successful! Output shape: {xvector_outputs[0].shape}")
    
    return True

def create_complete_inference_pipeline():
    """Create a complete inference pipeline using both ONNX models."""
    
    class CompleteVoiceFilterONNXPipeline:
        def __init__(self, main_onnx_path: str, xvector_onnx_path: str):
            import onnxruntime as ort
            self.main_session = ort.InferenceSession(main_onnx_path)
            self.xvector_session = ort.InferenceSession(xvector_onnx_path)
            self.stft_processor = STFTProcessor()
            
        def process_audio(self, mixed_audio: np.ndarray, reference_audio: np.ndarray) -> np.ndarray:
            """
            Process audio using both ONNX models.
            
            Args:
                mixed_audio: Mixed audio signal (16kHz)
                reference_audio: Reference speaker audio (16kHz)
                
            Returns:
                Enhanced audio signal
            """
            # Calculate speaker embedding from reference audio using ONNX XVector
            # Preprocess reference audio for XVector
            ref_audio_tensor = torch.from_numpy(reference_audio).unsqueeze(0).unsqueeze(0).float()
            
            # Run XVector inference
            xvector_inputs = {'audio': ref_audio_tensor.numpy()}
            speaker_embedding = self.xvector_session.run(['speaker_embedding'], xvector_inputs)[0]
            
            # Compute STFT of mixed audio
            spectrogram = self.stft_processor.compute_stft(mixed_audio)
            
            # Prepare inputs for main ONNX model
            spectrogram_tensor = torch.from_numpy(spectrogram).unsqueeze(0).float()
            
            # Run main model inference
            main_inputs = {
                'spectrogram': spectrogram_tensor.numpy(),
                'speaker_embedding': speaker_embedding
            }
            
            enhanced_spec = self.main_session.run(['enhanced_spectrogram'], main_inputs)[0]
            
            # Convert back to audio
            enhanced_audio = self.stft_processor.compute_istft(enhanced_spec[0])
            
            return enhanced_audio
    
    return CompleteVoiceFilterONNXPipeline

def optimize_onnx_models():
    """Optimize ONNX models for mobile deployment."""
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic
    
    print("Optimizing ONNX models for mobile deployment...")
    
    # Quantize main model
    print("Quantizing main model...")
    quantize_dynamic(
        model_input="voice_filter_main.onnx",
        model_output="voice_filter_main_quantized.onnx",
        weight_type=ort.quantization.QuantType.QUInt8
    )
    
    # Quantize XVector model
    print("Quantizing XVector model...")
    quantize_dynamic(
        model_input="voice_filter_xvector.onnx",
        model_output="voice_filter_xvector_quantized.onnx",
        weight_type=ort.quantization.QuantType.QUInt8
    )
    
    print("Model optimization completed!")
    return "voice_filter_main_quantized.onnx", "voice_filter_xvector_quantized.onnx"

if __name__ == "__main__":
    # Convert both models to ONNX
    main_onnx_path = convert_voice_filter_to_onnx()
    xvector_onnx_path = convert_xvector_to_onnx()
    
    # Test the ONNX models
    test_onnx_models()
    
    # Optimize models for mobile
    optimized_main, optimized_xvector = optimize_onnx_models()
    
    print("Conversion and optimization completed!")
    print(f"Main model: {main_onnx_path}")
    print(f"XVector model: {xvector_onnx_path}")
    print(f"Optimized main model: {optimized_main}")
    print(f"Optimized XVector model: {optimized_xvector}") 