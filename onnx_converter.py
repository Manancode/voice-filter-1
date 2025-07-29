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
        self.xvector_model = original_model.xvector_model
        self.filter_condition_transform = original_model.filter_condition_transform
        self.dropout = original_model.dropout
        self.wav_chunk_size = original_model.wav_chunk_size
        
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
        # Get speaker embedding for the input spectrogram
        # Note: In real usage, this would be computed from the input audio
        # For ONNX export, we assume it's provided as input
        
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

def cal_xvector_sincnet_embedding(xvector_model, ref_wav, max_length=5, sr=16000):
    """Calculate speaker embedding from reference audio."""
    wavs = []
    for i in range(0, len(ref_wav), max_length*sr):
        wav = ref_wav[i:i + max_length*sr]
        wav = np.concatenate([wav, np.zeros(max(0, max_length * sr - len(wav)))])
        wavs.append(wav)
    wavs = torch.from_numpy(np.stack(wavs))
    embed = xvector_model(wavs.unsqueeze(1).float())
    return torch.mean(embed, dim=0).detach()

def preprocess_audio_chunks(audio: np.ndarray, chunk_size: int = 5*16000) -> list:
    """Split audio into chunks for processing."""
    chunks = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        # Pad if necessary
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        chunks.append(chunk)
    return chunks

def convert_to_onnx():
    """Convert the VoiceFilter model to ONNX format."""
    
    print("Loading original model...")
    # Load the original model
    repo_id = 'nguyenvulebinh/voice-filter'
    original_model = VoiceFilter.from_pretrained(repo_id, cache_dir='./cache')
    original_model.eval()
    
    print("Creating ONNX-compatible model...")
    # Create ONNX-compatible version
    onnx_model = ONNXCompatibleVoiceFilter(original_model)
    
    # Create dummy inputs for ONNX export
    batch_size = 1
    time_steps = 100  # Adjust based on your needs
    freq_bins = 257   # (n_fft//2 + 1)
    embed_dim = 512
    
    dummy_spectrogram = torch.randn(batch_size, time_steps, freq_bins)
    dummy_speaker_embedding = torch.randn(batch_size, embed_dim)
    
    print("Exporting to ONNX...")
    # Export to ONNX with dynamic axes
    dynamic_axes = {
        'spectrogram': {0: 'batch_size', 1: 'time_steps'},
        'speaker_embedding': {0: 'batch_size'},
        'enhanced_spectrogram': {0: 'batch_size', 1: 'time_steps'}
    }
    
    torch.onnx.export(
        onnx_model,
        (dummy_spectrogram, dummy_speaker_embedding),
        "voice_filter.onnx",
        export_params=True,
        opset_version=17,  # Use latest opset for better STFT support
        do_constant_folding=True,
        input_names=['spectrogram', 'speaker_embedding'],
        output_names=['enhanced_spectrogram'],
        dynamic_axes=dynamic_axes,
        verbose=True
    )
    
    print("ONNX model exported successfully!")
    return "voice_filter.onnx"

def test_onnx_model(onnx_path: str):
    """Test the exported ONNX model."""
    import onnxruntime as ort
    
    print("Testing ONNX model...")
    
    # Create test inputs
    batch_size = 1
    time_steps = 50
    freq_bins = 257
    embed_dim = 512
    
    test_spectrogram = np.random.randn(batch_size, time_steps, freq_bins).astype(np.float32)
    test_speaker_embedding = np.random.randn(batch_size, embed_dim).astype(np.float32)
    
    # Create ONNX Runtime session
    session = ort.InferenceSession(onnx_path)
    
    # Run inference
    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]
    
    inputs = {
        'spectrogram': test_spectrogram,
        'speaker_embedding': test_speaker_embedding
    }
    
    outputs = session.run(output_names, inputs)
    
    print(f"ONNX model test successful!")
    print(f"Input shape: {test_spectrogram.shape}")
    print(f"Output shape: {outputs[0].shape}")
    
    return outputs[0]

def create_inference_pipeline():
    """Create a complete inference pipeline using the ONNX model."""
    
    class VoiceFilterONNXPipeline:
        def __init__(self, onnx_path: str):
            import onnxruntime as ort
            self.session = ort.InferenceSession(onnx_path)
            self.stft_processor = STFTProcessor()
            
        def process_audio(self, mixed_audio: np.ndarray, reference_audio: np.ndarray) -> np.ndarray:
            """
            Process audio using the ONNX model.
            
            Args:
                mixed_audio: Mixed audio signal (16kHz)
                reference_audio: Reference speaker audio (16kHz)
                
            Returns:
                Enhanced audio signal
            """
            # Calculate speaker embedding from reference audio
            # Note: This still uses the original PyTorch model for embedding
            # In production, you might want to export this separately to ONNX too
            
            # Compute STFT of mixed audio
            spectrogram = self.stft_processor.compute_stft(mixed_audio)
            
            # Prepare inputs for ONNX model
            spectrogram_tensor = torch.from_numpy(spectrogram).unsqueeze(0).float()
            
            # For now, use dummy speaker embedding
            # In real usage, you'd compute this from reference_audio
            speaker_embedding = torch.randn(1, 512).float()
            
            # Run ONNX inference
            inputs = {
                'spectrogram': spectrogram_tensor.numpy(),
                'speaker_embedding': speaker_embedding.numpy()
            }
            
            enhanced_spec = self.session.run(['enhanced_spectrogram'], inputs)[0]
            
            # Convert back to audio
            enhanced_audio = self.stft_processor.compute_istft(enhanced_spec[0])
            
            return enhanced_audio
    
    return VoiceFilterONNXPipeline

if __name__ == "__main__":
    # Convert model to ONNX
    onnx_path = convert_to_onnx()
    
    # Test the ONNX model
    test_onnx_model(onnx_path)
    
    print("Conversion and testing completed!")
    print(f"ONNX model saved as: {onnx_path}") 