import os
import sys
import queue
import threading
import time
import logging
import re

import numpy as np
import sherpa_onnx
import soundfile as sf
import sounddevice as sd

from ..utils.utils import resource_path
from .share_state import State

# =================== Global State ===================
class AudioState:
    """Centralized audio state management"""
    def __init__(self):
        self.buffer = queue.Queue(maxsize=200)  # Very large buffer size to prevent underruns
        self.started = False
        self.killed = False
        self.sample_rate = 16000
        self.first_message_time = None
        self.play_thread = None
        self.play_thread_started = False
        self.event = threading.Event()  # Signal to stop playback thread
        self.playback_finished_event = threading.Event()  # Notification for completed playback
        self.play_thread_lock = threading.Lock()
        self.buffer_low_threshold = 10  # Minimum buffer size to avoid underruns
        self.thread_priority_set = False

# Create global state instance
audio_state = AudioState()

# =================== Audio Callback Functions ===================
def generated_audio_callback(samples: np.ndarray, progress: float):
    """Callback for when audio is generated"""
    if audio_state.first_message_time is None:
        audio_state.first_message_time = time.time()
    
    try:
        # Add samples to buffer with 1 second timeout to prevent blocking
        audio_state.buffer.put(samples, timeout=1.0)
    except queue.Full:
        logging.warning("Audio buffer full, dropping some samples")
        # Try to put again after dropping the oldest sample if buffer is full
        try:
            audio_state.buffer.get_nowait()  # Remove oldest item
            audio_state.buffer.put(samples, timeout=0.5)
        except (queue.Empty, queue.Full):
            pass  # Give up if still having issues

    if not audio_state.started:
        logging.info("Start playing ...")
        audio_state.started = True
        State.pause_listening()  # Stop voice listening

    return 0 if audio_state.killed else 1

def play_audio_callback(outdata: np.ndarray, frames: int, cbtime, status: sd.CallbackFlags):
    """Callback for audio playback"""
    # Log status flags for debugging
    if status:
        # Only log severe errors to avoid spamming the log
        if status & sd.CallbackFlags.OVERFLOW:
            logging.warning("Audio buffer overflow")
        if status & sd.CallbackFlags.UNDERFLOW:
            logging.debug("Audio buffer underflow")  # Reduced to debug level to avoid log spam
    
    if audio_state.killed:
        audio_state.event.set()
        outdata.fill(0)
        return

    try:
        n = 0
        # Use a more efficient approach - fill the entire buffer at once when possible
        if audio_state.buffer.qsize() > 0:
            # Get the first sample block
            first_samples = audio_state.buffer.queue[0]
            
            if len(first_samples) >= frames:
                # If the first block is big enough, use it directly
                outdata[:, 0] = first_samples[:frames]
                
                # Keep remaining samples in the queue
                if len(first_samples) > frames:
                    audio_state.buffer.queue[0] = first_samples[frames:]
                else:
                    audio_state.buffer.get()  # Remove the used sample block
                
                return
        
        # Fallback to the previous block-by-block approach
        while n < frames:
            if audio_state.buffer.empty():
                # Fill remaining with zeros
                outdata[n:, 0] = 0
                
                if audio_state.started:
                    logging.info("Playback finished.")
                    State().resume_listening()
                    audio_state.started = False
                    audio_state.playback_finished_event.set()  # Notify playback completion
                
                return

            samples = audio_state.buffer.queue[0]
            remaining = frames - n
            sample_len = samples.shape[0]

            if remaining < sample_len:
                outdata[n:n+remaining, 0] = samples[:remaining]
                audio_state.buffer.queue[0] = samples[remaining:]
                n += remaining
            else:
                outdata[n:n+sample_len, 0] = samples
                audio_state.buffer.get()
                n += sample_len
    except Exception as e:
        logging.error(f"Error in audio callback: {e}")
        outdata.fill(0)  # Fill with silence on error

# =================== Audio Control Functions ===================
def play_audio():
    """Thread function for audio playback"""
    logging.info("Audio playback thread started.")
    
    # Settings to minimize underruns
    buffer_size = 4096  # Much larger buffer (previously 2048)
    extra_settings = {}
    
    # Platform-specific optimizations
    import platform
    if platform.system() == 'Linux':
        # On Linux, try to force ALSA to use larger buffers
        extra_settings['device'] = 'default'  # Use ALSA default device explicitly
        os.environ['ALSA_BUFFER_SIZE'] = '65536'  # Set very large ALSA buffer
    
    try:
        with sd.OutputStream(
            samplerate=audio_state.sample_rate,
            channels=1,
            dtype='float32',
            callback=play_audio_callback,
            blocksize=buffer_size,
            latency='high',
            prime_output_buffers_using_stream_callback=True,  # Fill buffers before starting
            **extra_settings
        ):
            # Pre-fill the buffer with silence to prevent initial underruns
            silence = np.zeros((buffer_size,), dtype=np.float32)
            for _ in range(3):  # Add several blocks of silence
                audio_state.buffer.put(silence)
                
            audio_state.event.wait()  # Wait for stop event
    except Exception as e:
        logging.error(f"Audio playback error: {e}")
    
    logging.info("Audio playback thread exited.")

def stop_playback():
    """Stop audio playback and wait for thread to finish"""
    audio_state.killed = True
    audio_state.event.set()
    
    # Wait for playback thread to finish if it exists
    if audio_state.play_thread and audio_state.play_thread.is_alive():
        audio_state.play_thread.join(timeout=1.0)  # Wait with timeout
        
    State().resume_listening()

def reset_playback():
    """Reset audio playback state"""
    stop_playback()  # Ensure playback is stopped first
    
    audio_state.buffer = queue.Queue()
    audio_state.started = False
    audio_state.killed = False
    audio_state.first_message_time = None
    audio_state.play_thread_started = False
    audio_state.event.clear()
    audio_state.playback_finished_event.clear()

# =================== Text-to-Speech Class ===================
class TextToSpeech:
    """Text-to-speech synthesis using sherpa-onnx"""
    
    def __init__(self, 
                 model_dir="sherpa/vits-icefall-zh-aishell3",  
                 backend="sherpa-onnx",
                 voice="af_alloy",   
                 speed=1.3,
                 priority_boost=False):
        """
        Initialize TTS engine
        
        Args:
            model_dir: Path to model directory
            backend: TTS backend engine
            voice: Voice ID
            speed: Speech speed factor
            priority_boost: Whether to boost thread priority for audio playback
        """
        self.backend = backend
        self.voice = voice
        self.speed = speed
        self.priority_boost = priority_boost

        self.model_dir = resource_path(model_dir)
        
        # Try to set process priority on Linux systems
        if priority_boost and platform.system() == 'Linux':
            try:
                import os
                os.nice(-10)  # Increase priority (-20 is highest, 19 is lowest)
                logging.info("Process priority boosted")
            except (ImportError, OSError, PermissionError) as e:
                logging.warning(f"Failed to boost process priority: {e}")

    def synthesize(self, text, output_file=None, wait_for_playback=False):
        """
        Synthesize text to speech
        
        Args:
            text: Text to synthesize
            output_file: Optional file path to save audio
            wait_for_playback: Whether to wait for playback to complete
            
        Returns:
            Audio object or None if failed
        """
        if self.backend == "sherpa-onnx":
            audio = self._synthesize_sherpa_onnx(text, output_file)
            
            # Wait for playback to complete if requested
            if wait_for_playback and audio:
                logging.info("Waiting for playback to complete...")
                audio_state.playback_finished_event.wait()
                logging.info("Playback completed.")
                
            return audio
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _find_model_files(self):
        """Find and classify model files in the model directory"""
        model_files = {
            "model": None,
            "vocoder": None,
            "lexicon": [],
            "tokens": None,
            "dict_dir": None,
            "data_dir": None,
            "rule_fsts": [],
            "kokoro_voices": None
        }

        for file in os.listdir(self.model_dir):
            file_path = os.path.join(self.model_dir, file)
            
            if re.match(r"model.*\.onnx$", file):
                model_files["model"] = file_path
            elif re.match(r"vocos.*\.onnx$", file):
                model_files["vocoder"] = file_path
            elif file == "voices.bin":
                model_files["kokoro_voices"] = file_path
            elif file in ["lexicon.txt", "lexicon-us-en.txt", "lexicon-zh.txt"]:
                model_files["lexicon"].append(file_path)
            elif file == "tokens.txt":
                model_files["tokens"] = file_path
            elif os.path.isdir(file_path) and file == "espeak-ng-data":
                model_files["data_dir"] = file_path
            elif os.path.isdir(file_path) and file == "dict":
                model_files["dict_dir"] = file_path
            elif file.endswith(".fst"):
                model_files["rule_fsts"].append(file_path)
                
        # Join lexicon paths
        if model_files["lexicon"]:
            model_files["lexicon"] = ",".join(model_files["lexicon"])
        else:
            model_files["lexicon"] = ""
            
        return model_files

    def _detect_provider(self):
        """Detect the best available provider for inference"""
        import platform
        import torch
        
        system = platform.system().lower()
        if system == "darwin":
            return "coreml"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _synthesize_sherpa_onnx(self, text, output_file=None):
        """Synthesize text using sherpa-onnx backend"""
        try:
            # Find model files
            model_files = self._find_model_files()
            
            if not model_files["model"]:
                raise FileNotFoundError("ONNX model file not found")
                
            # Ensure dict_dir is a string, not None
            if model_files.get("dict_dir") is None:
                model_files["dict_dir"] = ""
                
            # Ensure data_dir is a string, not None
            if model_files.get("data_dir") is None:
                model_files["data_dir"] = ""

            # Configure TTS
            provider = self._detect_provider()
            sid = 0
            num_threads = os.cpu_count()
            rule_fsts = ",".join(model_files["rule_fsts"]) if model_files["rule_fsts"] else ""

            # Create TTS config - following exact parameter order from sherpa_onnx
            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=model_files["model"],
                        lexicon=model_files["lexicon"],
                        tokens=model_files["tokens"],
                        data_dir=model_files.get("data_dir", ""),
                        dict_dir=model_files.get("dict_dir", ""),
                        noise_scale=0.667,
                        noise_scale_w=0.8,
                        length_scale=self.speed,
                    ),
                    provider="cpu",
                    debug=False,
                    num_threads=os.cpu_count(),  # Use all available cores
                ),
                rule_fsts=rule_fsts,
                max_num_sentences=1,
            )

            if not tts_config.validate():
                raise ValueError("Invalid TTS configuration")

            # Initialize TTS engine
            tts = sherpa_onnx.OfflineTts(tts_config)
            
            # Set global sample rate
            audio_state.sample_rate = tts.sample_rate
            audio_state.started = False
            
            # Generate audio
            start = time.time()
            audio = tts.generate(text, sid=sid, speed=self.speed, callback=generated_audio_callback)
            end = time.time()
            
            synthesis_time = end - start
            logging.info(f"Synthesis time: {synthesis_time:.3f} seconds")
            
            # Start playback thread if not already started
            with audio_state.play_thread_lock:
                if not audio_state.play_thread_started:
                    # Reset playback events before starting new thread
                    audio_state.event.clear()
                    audio_state.playback_finished_event.clear()
                    
                    audio_state.play_thread = threading.Thread(target=play_audio, daemon=True)
                    audio_state.play_thread.start()
                    audio_state.play_thread_started = True

            # Calculate performance metrics
            if len(audio.samples) > 0:
                audio_duration = len(audio.samples) / audio.sample_rate
                real_time_factor = synthesis_time / audio_duration
                logging.info(f"Audio duration: {audio_duration:.3f}s")
                logging.info(f"RTF: {synthesis_time:.3f}/{audio_duration:.3f} = {real_time_factor:.3f}")
                
                # Save to file if requested
                if output_file:
                    sf.write(output_file, audio.samples, audio.sample_rate)
                    logging.info(f"Audio saved to {output_file}")
                
                return audio
            else:
                logging.info("Generation failed, no audio produced")
                return None

        except Exception as e:
            logging.error(f"Synthesis failed: {e}")
            raise

# =================== Main Program ===================
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create TTS instance with boosted priority
    tts = TextToSpeech(
        backend="sherpa-onnx",
        model_dir="/path/to/sherpa-onnx/models",
        voice="zh-cn",
        speed=1.0,
        priority_boost=True,  # Try to boost process priority
    )

    try:
        # Synthesize text and wait for playback to complete
        tts.synthesize("你好，世界！", "output.wav", wait_for_playback=True)
    finally:
        # Ensure playback is stopped
        stop_playback()