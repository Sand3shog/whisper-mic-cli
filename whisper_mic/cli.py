#!/usr/bin/env python3
import os
import click
import torch
import speech_recognition as sr
from typing import Optional
from dotenv import load_dotenv

# Load .env if present, so OPENAI_API_KEY etc. get set
load_dotenv()

from whisper_mic.whisper_mic import WhisperMic

@click.command()
@click.option(
    "--model", default="base",
    type=click.Choice(["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]),
    help="Whisper model size"
)
@click.option(
    "--device", default=("cuda" if torch.cuda.is_available() else "cpu"),
    type=click.Choice(["cpu", "cuda", "mps"]),
    help="Compute device"
)
@click.option(
    "--impl", "implementation",
    default="whisper",
    type=click.Choice(["whisper", "faster_whisper", "whisperx", "openai", "google"]),
    help="Which transcription backend to use"
)
@click.option("--english",        is_flag=True, default=False, help="Use English-only model")
@click.option("--verbose",        is_flag=True, default=False, help="Emit verbose output")
@click.option("--energy",         default=300, type=int,   help="Static energy threshold")
@click.option("--dynamic_energy", is_flag=True, default=False, help="Enable dynamic energy")
@click.option("--pause",          default=0.8, type=float, help="Silence pause threshold")
@click.option("--hallucinate_threshold", default=400, type=int,
              help="Min amplitude to accept audio (higher = fewer false positives)")
@click.option("--save-file",      is_flag=True, default=False, help="Write transcripts to disk")
@click.option("--mic-index",      default=None, type=int, help="Microphone device index")
@click.option("--list-devices",   is_flag=True, default=False, help="List available mics and exit")
@click.option("--loop",           is_flag=True, default=False, help="Keep listening in a loop")
@click.option("--dictate",        is_flag=True, default=False, help="Type output into active window")
def main(
    model: str,
    device: str,
    implementation: str,
    english: bool,
    verbose: bool,
    energy: int,
    dynamic_energy: bool,
    pause: float,
    hallucinate_threshold: int,
    save_file: bool,
    mic_index: Optional[int],
    list_devices: bool,
    loop: bool,
    dictate: bool,
) -> None:
    # 1) List mics and exit
    if list_devices:
        names = sr.Microphone.list_microphone_names()
        click.echo("Available devices:")
        for idx, name in enumerate(names):
            click.echo(f"  {idx}: {name}")
        return

    # 2) If using OpenAI backend, ensure key is set
    if implementation == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise click.ClickException(
            "OPENAI_API_KEY not found in environment. "
            "Please set it or add it to a .env file."
        )

    # 3) Instantiate WhisperMic
    mic = WhisperMic(
        model=model,
        device=device,
        implementation=implementation,
        english=english,
        verbose=verbose,
        energy=energy,
        dynamic_energy=dynamic_energy,
        pause=pause,
        hallucinate_threshold=hallucinate_threshold,
        save_file=save_file,
        mic_index=mic_index,
    )

    try:
        if not loop:
            # Single-shot listen
            text = mic.listen()
            click.echo(f"You said: {text}")
        else:
            # Continuous loop (optionally typing out)
            mic.listen_loop(dictate=dictate, phrase_time_limit=2.0)
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user")
    finally:
        # Always clean up any open file handles
        if save_file and hasattr(mic, "file") and not mic.file.closed:
            mic.file.close()
        mic.close()

if __name__ == "__main__":
    main()
