import os
import time
import json
import queue
import tempfile
import threading
import platform
from typing import Optional

import torch
import numpy as np
import speech_recognition as sr
import pynput.keyboard

from whisper_mic.utils import get_logger
from rich.logging import RichHandler

class WhisperMic:
    """
    Real-time and file-based speech-to-text transcription using multiple backends:
      - whisper (OpenAI's open-source model)
      - faster_whisper (Intel-optimized)
      - whisperx (alternative implementation)
      - openai (Whisper API)
      - google (Google Cloud Speech-to-Text)
    """

    def __init__(
        self,
        model: str = "base",
        device: str = ("cuda" if torch.cuda.is_available() else "cpu"),
        english: bool = False,
        verbose: bool = False,
        energy: int = 300,
        pause: float = 2.0,
        dynamic_energy: bool = False,
        save_file: bool = False,
        model_root: str = "~/.cache/whisper",
        mic_index: Optional[int] = None,
        implementation: str = "whisper",
        hallucinate_threshold: float = 300.0,
    ):
        self.logger = get_logger("whisper_mic", "info")
        self.energy = energy
        self.pause = pause
        self.dynamic_energy = dynamic_energy
        self.save_file = save_file
        self.verbose = verbose
        self.english = english
        self.implementation = implementation.lower()
        self.hallucinate_threshold = hallucinate_threshold
        self.keyboard = pynput.keyboard.Controller()

        # Enforce CPU on macOS if necessary
        self.platform = platform.system().lower()
        if self.platform == "darwin" and device in ("cuda", "mps"):
            self.logger.warning("CUDA/mps not supported on macOS. Falling back to CPU.")
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Append `.en` for English-only models
        if self.english and model not in ("large", "large-v2", "large-v3"):
            model += ".en"

        model_root = os.path.expanduser(model_root)

        # Load audio model
        self.faster = False
        self.audio_model = self.__load_model(model, device, model_root)

        # Temp directory for intermediate files if saving
        self.temp_dir = tempfile.mkdtemp() if save_file else None

        # Queues for audio frames and transcription results
        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self.result_queue: queue.Queue[Optional[str]] = queue.Queue()

        self.break_threads = False
        self.mic_active = False
        self.banned_results = ["", " ", "\n", None]

        # Optional text file output
        if save_file:
            self.file = open("transcribed_text.txt", "w+", encoding="utf-8")

        # Set up microphone
        self.__setup_mic(mic_index)

    def __load_model(self, model, device, model_root):
        impl = self.implementation
        if impl == "whisper":
            import whisper
            m = whisper.load_model(model, download_root=model_root).to(device)
            return m

        elif impl == "faster_whisper":
            try:
                from faster_whisper import WhisperModel
                m = WhisperModel(model, download_root=model_root, device="auto", compute_type="int8")
                self.faster = True
                return m
            except ImportError:
                self.logger.error("faster_whisper not installed; falling back to whisper.")
                import whisper
                return whisper.load_model(model, download_root=model_root).to(device)

        elif impl == "whisperx":
            try:
                import whisperx
                m, _ = whisperx.load_model(model, device=device, download_root=model_root)
                return m
            except ImportError:
                self.logger.error("whisperx not installed. Install via `pip install whisperx`.")
                raise

        elif impl == "openai":
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            return None  # API-based

        elif impl == "google":
            from google.cloud import speech
            return speech.SpeechClient()

        else:
            raise ValueError(f"Unsupported implementation: {impl}")

    def __setup_mic(self, mic_index):
        if mic_index is None:
            self.logger.info("No mic index specified, using default device.")
        self.source = sr.Microphone(sample_rate=16000, device_index=mic_index)
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = self.energy
        self.recorder.pause_threshold = self.pause
        self.recorder.dynamic_energy_threshold = self.dynamic_energy
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)
        self.logger.info("Microphone setup complete.")

    def is_audio_loud_enough(self, frame: bytes) -> bool:
        audio_frame = np.frombuffer(frame, dtype=np.int16)
        return np.mean(np.abs(audio_frame)) > self.hallucinate_threshold

    def __preprocess(self, data: bytes):
        loud = self.is_audio_loud_enough(data)
        samples = np.frombuffer(data, np.int16).astype(np.float32) / 32768.0
        if self.implementation == "whisper":
            return torch.from_numpy(samples), loud
        else:
            return samples, loud

    def __get_all_audio(self, min_time: float = -1.0) -> bytes:
        buffer = bytes()
        start = time.time()
        got = False
        while not got or (min_time >= 0 and time.time() - start < min_time):
            while not self.audio_queue.empty():
                buffer += self.audio_queue.get()
                got = True
        return buffer

    def __record_load(self, _, audio: sr.AudioData) -> None:
        self.audio_queue.put_nowait(audio.get_raw_data())

    def __transcribe(self, data: Optional[bytes] = None) -> None:
        if data is None:
            data = self.__get_all_audio()
        audio_data, loud = self.__preprocess(data)
        if not loud:
            self.result_queue.put_nowait(None)
            return

        text = None

        if self.implementation == "faster_whisper":
            segments, _ = self.audio_model.transcribe(audio_data)
            text = "".join(s.text for s in segments)

        elif self.implementation == "whisper":
            kwargs = {"language": "english"} if self.english else {}
            result = self.audio_model.transcribe(audio_data, **kwargs, suppress_tokens="")
            text = result.get("text", "")

        elif self.implementation == "whisperx":
            # whisperx returns model output directly
            segments, _ = self.audio_model.transcribe(audio_data)
            text = "".join(s.text for s in segments)

        elif self.implementation == "openai":
            import openai
            temp_path = os.path.join(self.temp_dir or ".", "temp_audio.wav")
            with open(temp_path, "wb") as f:
                f.write(data)
            with open(temp_path, "rb") as f:
                text = openai.Audio.transcribe("whisper-1", f)["text"]

        elif self.implementation == "google":
            from google.cloud import speech
            audio = speech.RecognitionAudio(content=data)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="en-US"
            )
            response = self.audio_model.recognize(config=config, audio=audio)
            text = " ".join(r.alternatives[0].transcript for r in response.results)

        if text and text not in self.banned_results:
            self.result_queue.put_nowait(text)
            if self.save_file:
                self.file.write(text + "\n")
                self.save_transcription_json(text)
        else:
            self.result_queue.put_nowait(None)

    def listen(
        self,
        timeout: Optional[float] = None,
        phrase_time_limit: Optional[float] = None,
        try_again: bool = True
    ) -> Optional[str]:
        self.logger.info("Listening...")
        try:
            with self.source as mic:
                audio = self.recorder.listen(
                    source=mic, timeout=timeout, phrase_time_limit=phrase_time_limit
                )
            self.__record_load(0, audio)
            self.__transcribe()
        except sr.WaitTimeoutError:
            self.result_queue.put_nowait("Timeout: No speech detected.")
        except sr.UnknownValueError:
            self.result_queue.put_nowait("Speech could not be understood.")

        while True:
            result = self.result_queue.get()
            if result is None and try_again:
                return self.listen(timeout, phrase_time_limit, try_again)
            return result

    def record(
        self,
        duration: float = 2.0,
        offset: Optional[float] = None,
        try_again: bool = True
    ) -> Optional[str]:
        self.logger.info("Recording...")
        with self.source as mic:
            audio = self.recorder.record(source=mic, duration=duration, offset=offset)
        self.__record_load(0, audio)
        self.__transcribe()

        while True:
            result = self.result_queue.get()
            if result is None and try_again:
                return self.record(duration, offset, try_again)
            return result

    def listen_continuously(self, phrase_time_limit: Optional[float] = None):
        """
        Start non-blocking continuous transcription.
        Yields each transcription as it becomes available.
        """
        self.recorder.listen_in_background(self.source, self.__record_load, phrase_time_limit=phrase_time_limit)
        threading.Thread(target=self.__transcribe_forever, daemon=True).start()
        while True:
            yield self.result_queue.get()

    def __transcribe_forever(self):
        while not self.break_threads:
            self.__transcribe()

    def transcribe_file(self, file_path: str) -> Optional[str]:
        """
        Load and transcribe a pre-recorded audio file.
        """
        with open(file_path, "rb") as f:
            data = f.read()
        self.__transcribe(data)
        return self.get_latest_transcription()

    def save_transcription_json(self, text: str) -> None:
        """
        Append a JSON record of the transcription with timestamp.
        """
        record = {"text": text, "timestamp": time.time()}
        with open("transcription_log.json", "a", encoding="utf-8") as f:
            json.dump(record, f)
            f.write("\n")

    def start_background_recording(self, interval: float = 2.0) -> None:
        """
        Record and transcribe in the background every `interval` seconds.
        """
        def _loop():
            self.mic_active = True
            while not self.break_threads:
                result = self.record(duration=interval, try_again=False)
                if result:
                    self.logger.info(f"[Background] {result}")
                time.sleep(0.1)
            self.mic_active = False

        threading.Thread(target=_loop, daemon=True).start()
        self.logger.info("Background recording started.")

    def stop_background_recording(self) -> None:
        """Signal the background thread to stop."""
        self.break_threads = True
        self.logger.info("Background recording stopping.")

    def get_latest_transcription(self) -> Optional[str]:
        """Non-blocking fetch of the most recent result."""
        return self.result_queue.get_nowait() if not self.result_queue.empty() else None

    def is_active(self) -> bool:
        """Return whether the mic is currently active."""
        return self.mic_active

    def close(self) -> None:
        """Clean up file handles and threads."""
        self.break_threads = True
        if self.save_file and hasattr(self, "file") and not self.file.closed:
            self.file.close()
        self.logger.info("WhisperMic closed.")

    def __del__(self):
        self.close()
