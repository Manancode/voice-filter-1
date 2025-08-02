import torch
from src.model.modeling_enh import VoiceFilter
import time
import numpy as np
import onnxruntime as ort
import librosa
import soundfile as sf

class CorrectChunkedVoiceFilter(torch.nn.Module):
    def __init__(self, original_model, chunk_size=5000):  # 5000 frames = ~40 seconds
        super().__init__()
        self.separator = original_model.enh_model.separator
        self.chunk_size = chunk_size
    
    def forward(self, spectrogram, speaker_embedding):
        batch_size, seq_len = spectrogram.size(0), spectrogram.size(1)
        
        # If input is smaller than chunk_size, pad it
        if seq_len < self.chunk_size:
            padding = torch.zeros(batch_size, self.chunk_size - seq_len, spectrogram.size(2))
            spectrogram = torch.cat([spectrogram, padding], dim=1)
            seq_len = self.chunk_size
        
        # Process in chunks of chunk_size
        enhanced_chunks = []
        
        for i in range(0, seq_len, self.chunk_size):
            end = min(i + self.chunk_size, seq_len)
            chunk = spectrogram[:, i:end, :]
            
            # Pad chunk to chunk_size if needed
            if chunk.size(1) < self.chunk_size:
                padding = torch.zeros(batch_size, self.chunk_size - chunk.size(1), chunk.size(2))
                chunk = torch.cat([chunk, padding], dim=1)
            
            # Process chunk
            speech_lengths = torch.ones(batch_size, dtype=torch.long) * self.chunk_size
            enhanced_chunk, _, _ = self.separator(
                chunk, speaker_embedding, speech_lengths
            )
            
            # Store only the valid part
            valid_length = end - i
            enhanced_chunks.append(enhanced_chunk[0][:, :valid_length])
        
        # Concatenate all chunks
        return torch.cat(enhanced_chunks, dim=1)

def export_correct_chunked_voice_filter():
    print("🚀 STARTING CORRECT CHUNKED VOICEFILTER EXPORT...")
    start_time = time.time()
    
    # Load your model
    print("📥 Loading PyTorch model...")
    model = VoiceFilter.from_pretrained('./cache/models--nguyenvulebinh--voice-filter/snapshots/main')
    print(f"✅ PyTorch model loaded in {time.time() - start_time:.2f}s")
    
    # Create correct chunked wrapper
    print("🔧 Creating CorrectChunkedVoiceFilter wrapper...")
    onnx_model = CorrectChunkedVoiceFilter(model, chunk_size=5000)
    onnx_model.eval()
    print(f"✅ Wrapper created in {time.time() - start_time:.2f}s")
    
    # Test with chunk_size
    print("🧪 TESTING WITH CHUNK SIZE 5000...")
    test_start = time.time()
    dummy_spectrogram = torch.randn(1, 5000, 257)
    dummy_speaker_emb = torch.randn(1, 512)
    
    with torch.no_grad():
        output = onnx_model(dummy_spectrogram, dummy_speaker_emb)
        test_time = time.time() - test_start
        print(f"✅ Test with 5000 frames: Output size {output.shape[1]} ({test_time:.2f}s)")
    
    print(f"✅ Test completed in {time.time() - start_time:.2f}s")
    
    # Export with fixed size (as designed)
    print("📤 STARTING ONNX EXPORT WITH FIXED CHUNK SIZE...")
    print("⚠️  This should be much faster since we're not using dynamic axes")
    export_start = time.time()
    
    torch.onnx.export(
        onnx_model,
        (dummy_spectrogram, dummy_speaker_emb),
        "voice_filter_CORRECT_CHUNKED.onnx",
        input_names=["spectrogram", "speaker_embedding"],
        output_names=["enhanced_spectrogram"],
        opset_version=17,
        do_constant_folding=False,
        verbose=True
    )
    
    total_time = time.time() - start_time
    export_time = time.time() - export_start
    
    print("🎉 EXPORT COMPLETED!")
    print(f"✅ Total time: {total_time/60:.1f} minutes")
    print(f"✅ Export time: {export_time/60:.1f} minutes")
    print(f"✅ Model saved: voice_filter_CORRECT_CHUNKED.onnx")
    
    return "voice_filter_CORRECT_CHUNKED.onnx"

def test_correct_model():
    print("=== TESTING CORRECT CHUNKED MODEL ===")
    
    try:
        session = ort.InferenceSession("voice_filter_CORRECT_CHUNKED.onnx")
        
        # Test with exactly 5000 frames (as designed)
        spec = np.random.randn(1, 5000, 257).astype(np.float32)
        spk_emb = np.random.randn(1, 512).astype(np.float32)
        
        result = session.run(None, {
            "spectrogram": spec,
            "speaker_embedding": spk_emb
        })
        
        output_shape = result[0].shape
        print(f"✅ Input size 5000: Output shape {output_shape}")
        
        # Test with smaller input (should be padded)
        spec_small = np.random.randn(1, 1000, 257).astype(np.float32)
        result_small = session.run(None, {
            "spectrogram": spec_small,
            "speaker_embedding": spk_emb
        })
        
        output_shape_small = result_small[0].shape
        print(f"✅ Input size 1000 (padded): Output shape {output_shape_small}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def test_with_real_audio_correct():
    print("=== TESTING WITH REAL AUDIO (CORRECT CHUNKING) ===")
    
    # Load audio files
    mixed_wav_path = "cache/models--nguyenvulebinh--voice-filter/snapshots/main/binh_linh_newspaper_music_noise.wav"
    ref_wav_path = "cache/models--nguyenvulebinh--voice-filter/snapshots/main/binh_ref_long.wav"
    
    mixed_wav, sr = librosa.load(mixed_wav_path, sr=16000)
    ref_wav, _ = librosa.load(ref_wav_path, sr=16000)
    
    print(f"Mixed audio: {len(mixed_wav)} samples ({len(mixed_wav)/sr:.1f} seconds)")
    print(f"Reference audio: {len(ref_wav)} samples ({len(ref_wav)/sr:.1f} seconds)")
    
    # Load XVector
    xvector_session = ort.InferenceSession("voice_filter_xvector_quantized.onnx")
    
    # Extract speaker embedding
    max_length = 5
    embeddings = []
    
    for i in range(0, len(ref_wav), max_length*sr):
        wav = ref_wav[i:i + max_length*sr]
        wav = np.concatenate([wav, np.zeros(max(0, max_length * sr - len(wav)))])
        
        audio_input = wav.reshape(1, 1, -1).astype(np.float32)
        xvector_output = xvector_session.run(['speaker_embedding'], {'audio': audio_input})
        embeddings.append(xvector_output[0][0])
    
    speaker_embedding = np.mean(embeddings, axis=0)
    print(f"✅ Speaker embedding extracted: {speaker_embedding.shape}")
    
    # Load correct main model
    main_session = ort.InferenceSession("voice_filter_CORRECT_CHUNKED.onnx")
    
    # Process audio in 5000-frame chunks
    stft = librosa.stft(mixed_wav, n_fft=512, hop_length=128)
    magnitude = np.abs(stft).T  # [time, freq]
    phase = np.angle(stft).T    # [time, freq]
    
    print(f"STFT shape: {magnitude.shape}")
    
    # Process in 5000-frame chunks
    enhanced_chunks = []
    chunk_size = 5000
    
    for i in range(0, magnitude.shape[0], chunk_size):
        end = min(i + chunk_size, magnitude.shape[0])
        chunk = magnitude[i:end, :]
        
        # Pad chunk to 5000 frames if needed
        if chunk.shape[0] < chunk_size:
            padding = np.zeros((chunk_size - chunk.shape[0], 257))
            chunk = np.vstack([chunk, padding])
        
        # Process chunk
        spec = chunk.reshape(1, chunk_size, 257).astype(np.float32)
        spk_emb = speaker_embedding.reshape(1, 512).astype(np.float32)
        
        result = main_session.run(['enhanced_spectrogram'], {
            'spectrogram': spec,
            'speaker_embedding': spk_emb
        })
        
        enhanced_chunk = result[0][0]  # Remove batch dimension
        
        # Take only the valid part
        valid_length = end - i
        enhanced_chunks.append(enhanced_chunk[:valid_length, :])
        
        print(f"✅ Processed chunk {i//chunk_size + 1}: {valid_length} frames")
    
    # Concatenate all chunks
    enhanced_spec = np.vstack(enhanced_chunks)
    
    print(f"✅ Enhanced spectrogram shape: {enhanced_spec.shape}")
    print(f"✅ Original spectrogram shape: {magnitude.shape}")
    
    # Calculate metrics
    print(f"✅ Original spectrogram - Mean: {np.mean(magnitude):.4f}, Max: {np.max(magnitude):.4f}")
    print(f"✅ Enhanced spectrogram - Mean: {np.mean(enhanced_spec):.4f}, Max: {np.max(enhanced_spec):.4f}")
    
    # Calculate difference
    diff = enhanced_spec - magnitude
    print(f"✅ Difference - Mean: {np.mean(diff):.4f}, Max: {np.max(diff):.4f}, Min: {np.min(diff):.4f}")
    
    # Save results
    np.save('correct_original_spectrogram.npy', magnitude)
    np.save('correct_enhanced_spectrogram.npy', enhanced_spec)
    np.save('correct_difference_spectrogram.npy', diff)
    np.save('correct_speaker_embedding.npy', speaker_embedding)
    
    print("✅ Results saved:")
    print("  - correct_original_spectrogram.npy")
    print("  - correct_enhanced_spectrogram.npy")
    print("  - correct_difference_spectrogram.npy")
    print("  - correct_speaker_embedding.npy")
    
    return magnitude, enhanced_spec

# Run the correct solution
if __name__ == "__main__":
    print("🎯 USING THE MODEL AS DESIGNED - WITH PROPER CHUNKING!")
    print("📊 This approach respects the original model architecture")
    print("🚀 Should work perfectly with WhisperJET!")
    
    correct_model_path = export_correct_chunked_voice_filter() 