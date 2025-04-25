# 🎙️ Whisper Mic

Real-time speech-to-text transcription from your microphone using [OpenAI's Whisper](https://github.com/openai/whisper) model. Built for flexibility, ease of use, and continuous voice capture.


# Key highlights:
- **Real-time transcription** from your system microphone
- **File-based transcription** support for pre-recorded audio
- Multiple transcription backends:
  - **whisper** (OpenAI’s open-source model)
  - **faster_whisper** (Intel‑optimized quantized model)
  - **whisperx** (alternative high‑accuracy implementation)
  - **openai** (Whisper API)
  - **google** (Google Cloud Speech-to-Text)
- Configurable energy thresholds, silence detection, dynamic adjustment
- Optional logging to text and JSON for auditing and downstream processing
- Cross-platform support with automatic fallback on macOS
- Lightweight dependencies and easy installation via pip or source

---

## 📦 Installation

```bash
# Install from PyPI
pip install whisper-mic

# Or install the latest version from source:
git clone https://github.com/Sand3shog/whisper-mic-cli.git
cd whisper-mic-cli
pip install .
```

Ensure that you have:
- Python 3.9+
- FFmpeg (`ffmpeg` executable) in PATH for WAV conversions
- (Optional) Google Cloud credentials or OpenAI API key in env variables

---

## 🛠️ Quickstart

### Live Transcription

```bash
# Default uses local whisper model on CPU/GPU
whisper_mic --model small --implementation whisper --save-file
```

### Transcribe an Audio File

```bash
whisper_mic transcribe-file /path/to/audio.wav
```


## 🧩 Extending & Contributing

Contributions are welcome! You can help by:

- Adding new transcription backends (e.g. Azure, AssemblyAI)
- Improving performance (quantized models, batching strategies)
- Building a graphical or web-based front-end
- Implementing speaker diarization and punctuation
- Improving language and accent support

Please open issues or pull requests on the GitHub repo.

---

## 🔮 Future Roadmap

- **Websocket & REST API** for remote applications
- **Multi-language support** beyond English
- **GUI dashboard** with live subtitles
- **Noise suppression** and audio enhancement
- **Speaker diarization** (who said what)
- **Timestamp alignment** in JSON output
- **Integration with meeting platforms** (Zoom, Teams)

---


