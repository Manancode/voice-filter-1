import torch
from src.model.modeling_enh import VoiceFilter
import time
import numpy as np
import onnxruntime as ort
import librosa
import soundfile as sf

class PerfectVoiceFilter(torch.nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.separator = original_model.enh_model.separator
    
    def forward(self, spectrogram, speaker_embedding):
        batch_size, seq_len = spectrogram.size(0), spectrogram.size(1)
        
        # Process entire spectrogram at once (no chunking)
        speech_lengths = torch.ones(batch_size, dtype=torch.long) * seq_len
        enhanced_spec, _, _ = self.separator(
            spectrogram, speaker_embedding, speech_lengths
        )
        return enhanced_spec[0]  # Return full enhanced spectrogram

def export_perfect_voice_filter():
    print("🚀 STARTING PERFECT VOICEFILTER EXPORT...")
    start_time = time.time()
    
    # Load your model
    print("📥 Loading PyTorch model...")
    model = VoiceFilter.from_pretrained('./cache/models--nguyenvulebinh--voice-filter/snapshots/main')
    print(f"✅ PyTorch model loaded in {time.time() - start_time:.2f}s")
    
    # Create perfect wrapper
    print("🔧 Creating PerfectVoiceFilter wrapper...")
    onnx_model = PerfectVoiceFilter(model)
    onnx_model.eval()
    print(f"✅ Wrapper created in {time.time() - start_time:.2f}s")
    
    # Test with various sizes
    print("🧪 TESTING UNLIMITED INPUT SIZES...")
    test_sizes = [100, 500, 1000, 2000, 5000]
    
    for i, size in enumerate(test_sizes):
        test_start = time.time()
        dummy_spectrogram = torch.randn(1, size, 257)
        dummy_speaker_emb = torch.randn(1, 512)
        
        with torch.no_grad():
            output = onnx_model(dummy_spectrogram, dummy_speaker_emb)
            test_time = time.time() - test_start
            print(f"✅ Test {i+1}/5 - Input size {size}: Output size {output.shape[1]} ({test_time:.2f}s)")
    
    print(f"✅ All tests completed in {time.time() - start_time:.2f}s")
    
    # Export with dynamic axes for unlimited input
    print("📤 STARTING ONNX EXPORT WITH DYNAMIC AXES...")
    print("⚠️  This may take 30-60 minutes for large models with dynamic axes")
    export_start = time.time()
    
    torch.onnx.export(
        onnx_model,
        (dummy_spectrogram, dummy_speaker_emb),
        "voice_filter_CHUNKED.onnx",
        input_names=["spectrogram", "speaker_embedding"],
        output_names=["enhanced_spectrogram"],
        dynamic_axes={
            'spectrogram': {1: 'sequence_length'},
            'enhanced_spectrogram': {1: 'sequence_length'}
        },
        opset_version=17,
        do_constant_folding=False,
        verbose=True
    )
    
    total_time = time.time() - start_time
    export_time = time.time() - export_start
    
    print("🎉 EXPORT COMPLETED!")
    print(f"✅ Total time: {total_time/60:.1f} minutes")
    print(f"✅ Export time: {export_time/60:.1f} minutes")
    print(f"✅ Model saved: voice_filter_CHUNKED.onnx")
    
    return "voice_filter_CHUNKED.onnx"

def test_model_io():
    print("=== TESTING NEW MODEL INPUT/OUTPUT SPECIFICATIONS ===")
    
    try:
        session = ort.InferenceSession("voice_filter_CHUNKED.onnx")
        
        # Get input details
        inputs = session.get_inputs()
        print(f"Inputs ({len(inputs)}):")
        for i, input_info in enumerate(inputs):
            print(f"  {i+1}. Name: {input_info.name}")
            print(f"     Shape: {input_info.shape}")
            print(f"     Type: {input_info.type}")
        
        # Get output details
        outputs = session.get_outputs()
        print(f"Outputs ({len(outputs)}):")
        for i, output_info in enumerate(outputs):
            print(f"  {i+1}. Name: {output_info.name}")
            print(f"     Shape: {output_info.shape}")
            print(f"     Type: {output_info.type}")
            
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")
    
    return True

def test_variable_input_sizes():
    print("=== TESTING VARIABLE INPUT SIZES ===")
    
    session = ort.InferenceSession("voice_filter_CHUNKED.onnx")
    
    # Test different spectrogram sizes
    test_sizes = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    
    for size in test_sizes:
        try:
            # Create test inputs
            spec = np.random.randn(1, size, 257).astype(np.float32)
            spk_emb = np.random.randn(1, 512).astype(np.float32)
            
            # Run inference
            result = session.run(None, {
                "spectrogram": spec,
                "speaker_embedding": spk_emb
            })
            
            output_shape = result[0].shape
            print(f"✅ Input size {size}: Output shape {output_shape}")
            
        except Exception as e:
            print(f"❌ Input size {size}: {e}")
    
    return True

def test_with_real_audio():
    print("=== TESTING WITH REAL AUDIO ===")
    
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
    
    # Load new main model
    main_session = ort.InferenceSession("voice_filter_CHUNKED.onnx")
    
    # Process entire audio at once
    stft = librosa.stft(mixed_wav, n_fft=512, hop_length=128)
    magnitude = np.abs(stft).T  # [time, freq]
    phase = np.angle(stft).T    # [time, freq]
    
    print(f"STFT shape: {magnitude.shape}")
    
    # Process entire spectrogram
    spec = magnitude.reshape(1, magnitude.shape[0], 257).astype(np.float32)
    spk_emb = speaker_embedding.reshape(1, 512).astype(np.float32)
    
    result = main_session.run(['enhanced_spectrogram'], {
        'spectrogram': spec,
        'speaker_embedding': spk_emb
    })
    
    enhanced_spec = result[0][0]  # Remove batch dimension
    
    print(f"✅ Enhanced spectrogram shape: {enhanced_spec.shape}")
    print(f"✅ Original spectrogram shape: {magnitude.shape}")
    
    # Calculate metrics
    print(f"✅ Original spectrogram - Mean: {np.mean(magnitude):.4f}, Max: {np.max(magnitude):.4f}")
    print(f"✅ Enhanced spectrogram - Mean: {np.mean(enhanced_spec):.4f}, Max: {np.max(enhanced_spec):.4f}")
    
    # Calculate difference
    diff = enhanced_spec - magnitude
    print(f"✅ Difference - Mean: {np.mean(diff):.4f}, Max: {np.max(diff):.4f}, Min: {np.min(diff):.4f}")
    
    # Save results
    np.save('new_original_spectrogram.npy', magnitude)
    np.save('new_enhanced_spectrogram.npy', enhanced_spec)
    np.save('new_difference_spectrogram.npy', diff)
    np.save('new_speaker_embedding.npy', speaker_embedding)
    
    print("✅ Results saved:")
    print("  - new_original_spectrogram.npy")
    print("  - new_enhanced_spectrogram.npy")
    print("  - new_difference_spectrogram.npy")
    print("  - new_speaker_embedding.npy")
    
    return magnitude, enhanced_spec

def test_performance():
    print("=== TESTING MODEL PERFORMANCE ===")
    
    # Load model
    start_time = time.time()
    session = ort.InferenceSession("voice_filter_CHUNKED.onnx")
    load_time = time.time() - start_time
    print(f"✅ Model load time: {load_time:.3f} seconds")
    
    # Test inference speed with different sizes
    test_sizes = [100, 500, 1000, 2000, 5000]
    
    for size in test_sizes:
        spec = np.random.randn(1, size, 257).astype(np.float32)
        spk_emb = np.random.randn(1, 512).astype(np.float32)
        
        # Warm up
        for _ in range(3):
            session.run(None, {"spectrogram": spec, "speaker_embedding": spk_emb})
        
        # Test inference time
        times = []
        for _ in range(5):
            start_time = time.time()
            result = session.run(None, {"spectrogram": spec, "speaker_embedding": spk_emb})
            inference_time = time.time() - start_time
            times.append(inference_time)
        
        avg_time = np.mean(times)
        print(f"✅ Size {size}: {avg_time:.4f}s average inference time")
    
    return True

def monitor_export():
    print("📊 EXPORT PROGRESS MONITORING:")
    print("⏰ Expected timeline:")
    print("   - 0-10 min: Model analysis")
    print("   - 10-30 min: Graph optimization")
    print("   - 30-60 min: Dynamic axes processing")
    print("   - 60+ min: Final optimization")
    print("")
    print("💡 Tips:")
    print("   - Keep terminal open")
    print("   - Monitor CPU usage")
    print("   - Check disk space")
    print("   - Don't interrupt process")

# Run with monitoring
if __name__ == "__main__":
    monitor_export()
    chunked_model_path = export_perfect_voice_filter() 