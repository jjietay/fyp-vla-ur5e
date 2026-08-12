"""speech.py

It takes a press of Enter and gives you what the person said, as a string.

Push to talk: Enter opens the microphone, Enter closes it, and the complete clip
goes to `faster-whisper` in one piece. Capturing a whole utterance this way
sidesteps endpointing, voice activity detection and partial hypotheses entirely,
which matters because a robotics lab is noisy and none of that is what this
project is about.

Lives in `shared/` because ASR is a CONTROLLED VARIABLE. One microphone, one
model, one transcript, handed to Architecture A and to Architecture B unchanged.
Normalising the transcript for B would insert an LLM into B and it would stop
being end to end, so the raw text goes to both and the asymmetry gets measured
instead of patched.

Note what this module does NOT do: it never touches a pipeline. `ArchitectureA.run`
takes a plain string and has no idea how it arrived. The caller records, then
hands over text. Keeping ASR outside both architectures is what makes it a shared
variable rather than a feature one of them owns.

Nothing in here cleans up the transcript. No punctuation stripping, no filler word
removal, no lowercasing. Disfluent speech is data.
"""
from __future__ import annotations

import json
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from fyp.shared.helpers.config import get_config, resolve


@dataclass(frozen=True)
class Transcript:
    """One utterance: what was said, how long it took, and where the audio went."""
    text: str
    duration_s: float          # length of the recorded audio
    latency_s: float           # wall clock spent inside Whisper
    language: str
    model: str
    started_at: str            # ISO 8601, local time
    audio_path: Path | None = None
    segments: list[dict] = field(default_factory=list)

    def __bool__(self) -> bool:
        """An empty transcript is falsey, so `if not t:` catches silence."""
        return bool(self.text.strip())


class NoAudioCaptured(RuntimeError):
    """Raised when the clip was too short to be speech, rather than returning ""."""


def list_input_devices() -> str:
    """
    It takes nothing and gives you the available microphones as printable text.

    The first thing to run when a recording comes back silent. A USB headset
    often does not become the system default just by being plugged in.
    """
    sd = _import_sounddevice()
    lines = [f"default input: {sd.default.device[0]}"]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            lines.append(f"  [{i}] {d['name']}  ({d['max_input_channels']} ch, "
                         f"{d['default_samplerate']:.0f} Hz)")
    return "\n".join(lines)


def record_push_to_talk(samplerate: int = 16000, input_device=None,
                        prompt: str = "Enter to speak") -> np.ndarray:
    """
    It takes a keypress and gives you the recorded audio as mono float32.

    Enter starts, Enter stops. Deliberately not hold-to-talk: a held key needs a
    real display session, which breaks the moment you drive the cell over SSH.

    Recording at Whisper's own 16 kHz means no resampling happens anywhere, so
    there is no quiet quality loss and no scipy dependency.
    """
    sd = _import_sounddevice()
    blocks: list[np.ndarray] = []
    overflows = 0

    def _callback(indata, _frames, _time_info, status):
        nonlocal overflows
        if status:
            # counted rather than printed: printing from an audio callback can
            # itself cause the next overflow
            overflows += 1
        # .copy() is not optional. sounddevice hands the same buffer back every
        # callback, so appending it directly gives you N references to the last
        # block of audio and a recording that is pure tail.
        blocks.append(indata.copy())

    try:
        input(f"\n  {prompt} > ")
    except EOFError:
        print()
        return np.zeros(0, dtype=np.float32)

    with sd.InputStream(samplerate=samplerate, channels=1, dtype="float32",
                        device=input_device, callback=_callback):
        print("  RECORDING, Enter to stop > ", end="", flush=True)
        try:
            input()
        except EOFError:
            print()

    if overflows:
        print(f"  note: {overflows} audio buffer overflow(s), the clip may have gaps")
    if not blocks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(blocks, axis=0)[:, 0]


class SpeechToText:
    """
    It takes the speech block from config and gives you a microphone you can ask
    for a transcript.

    Built once and reused. `large-v3` takes several seconds to load, and paying
    that per utterance would dominate the latency numbers and stall a live demo
    between commands:

        with SpeechToText() as ears:
            while True:
                print(ears.listen().text)
    """

    def __init__(self, cfg: dict | None = None, input_device=None,
                 model: str | None = None, device: str | None = None):
        cfg = cfg or get_config()
        s = cfg["speech"]

        self.model_name = model or s["model"]
        self.samplerate = int(s["samplerate"])
        self.language = s["language"]
        self.beam_size = int(s["beam_size"])
        self.min_duration_s = float(s["min_duration_s"])
        self.max_duration_s = float(s["max_duration_s"])
        self.save_audio = bool(s["save_audio"])
        self.log_dir = resolve(s["log_dir"])
        self.input_device = input_device

        self.device = device or s["device"]
        self.compute_type = _compute_type_for(self.device, s["compute_type"])

        WhisperModel = _import_faster_whisper()
        t0 = time.perf_counter()
        # Loaded eagerly so the wait happens at startup rather than in the gap
        # after someone has just spoken to the robot.
        self._model = WhisperModel(self.model_name, device=self.device,
                                   compute_type=self.compute_type)
        self.load_s = time.perf_counter() - t0

    # ------------------------------------------------------------------ use

    def listen(self, prompt: str = "Enter to speak") -> Transcript:
        """
        It takes a press of Enter and gives you a Transcript of what was said.

        The one method the rest of the project calls.
        """
        started_at = datetime.now().isoformat(timespec="milliseconds")
        audio = record_push_to_talk(self.samplerate, self.input_device, prompt)
        return self.transcribe(audio, started_at=started_at)

    def transcribe(self, audio: np.ndarray, started_at: str | None = None) -> Transcript:
        """
        It takes mono 16 kHz float32 audio and gives you a Transcript.

        Split out from `listen` so the recogniser can be tested against a wav
        file with no microphone attached, which is the only way any of this is
        testable off the lab machine.
        """
        started_at = started_at or datetime.now().isoformat(timespec="milliseconds")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration_s = len(audio) / self.samplerate

        if duration_s < self.min_duration_s:
            raise NoAudioCaptured(
                f"got {duration_s:.2f}s of audio, below min_duration_s="
                f"{self.min_duration_s}. Either nothing was said, or the "
                f"microphone is not the one you think it is. Check:\n"
                f"{list_input_devices()}"
            )

        if duration_s > self.max_duration_s:
            print(f"  note: {duration_s:.1f}s clip truncated to {self.max_duration_s}s")
            audio = audio[:int(self.max_duration_s * self.samplerate)]
            duration_s = self.max_duration_s

        t0 = time.perf_counter()
        segments, info = self._model.transcribe(audio, language=self.language,
                                                beam_size=self.beam_size)
        # faster-whisper returns a generator and does the actual work lazily, so
        # nothing has run until this list() and timing it before here reads zero.
        seg_list = [{"start": sg.start, "end": sg.end, "text": sg.text} for sg in segments]
        latency_s = time.perf_counter() - t0

        text = "".join(sg["text"] for sg in seg_list).strip()

        audio_path = self._write_audio(audio, started_at) if self.save_audio else None
        t = Transcript(text=text, duration_s=duration_s, latency_s=latency_s,
                       language=getattr(info, "language", self.language or "?"),
                       model=self.model_name, started_at=started_at,
                       audio_path=audio_path, segments=seg_list)
        self._log(t)
        return t

    # -------------------------------------------------------------- logging

    def _write_audio(self, audio: np.ndarray, started_at: str) -> Path:
        """It takes float32 audio and gives you the path of the wav it wrote."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # milliseconds, not seconds: two utterances in the same second would
        # otherwise share a filename and the second would silently overwrite the
        # first, destroying exactly the audio evidence this exists to keep.
        stamp = started_at.replace(":", "").replace("-", "").replace(".", "_")
        path = self.log_dir / f"{stamp}.wav"
        # float32 [-1, 1] to int16 PCM. Clipped first, because a loud speaker
        # into a hot gain stage overflows and wraps to the opposite polarity,
        # which sounds like violent noise rather than like distortion.
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.samplerate)
            w.writeframes(pcm.tobytes())
        return path

    def _log(self, t: Transcript) -> None:
        """
        It takes a Transcript and appends it to the speech log.

        W2b asks for every transcript kept with its audio. When a trial fails for
        a reason that turns out to be mishearing, this is the only record that
        distinguishes "the architecture failed" from "the microphone did", and
        that distinction decides whether a trial counts.
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)
        row = {"started_at": t.started_at, "text": t.text,
               "duration_s": round(t.duration_s, 3),
               "latency_s": round(t.latency_s, 3),
               "language": t.language, "model": t.model,
               "audio": t.audio_path.name if t.audio_path else None}
        with open(self.log_dir / "transcripts.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

    # ------------------------------------------------------------ lifecycle

    def close(self) -> None:
        self._model = None

    def __enter__(self) -> "SpeechToText":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ------------------------------------------------------------------ helpers

def _compute_type_for(device: str, configured: str) -> str:
    """
    It takes the device and the configured precision and gives you one that
    actually runs there.

    float16 on CUDA is mandatory, not a preference: CTranslate2's default int8
    path crashes with CUBLAS_STATUS_NOT_SUPPORTED on sm_120 (Blackwell, the
    RTX 50 series), because its int8 tensor cores need padding older builds do
    not emit. See the Speech Stack note.

    CTranslate2 has no float16 CPU kernel at all, so asking for it there fails
    at load with a less obvious message. Fall back rather than crash, since the
    only reason to be on CPU is testing away from the lab machine.
    """
    if device == "cpu" and configured == "float16":
        print("  note: float16 is CUDA-only in CTranslate2, using float32 on cpu")
        return "float32"
    return configured


def _import_sounddevice():
    try:
        import sounddevice as sd
    except OSError as e:
        raise RuntimeError(
            "sounddevice imported but PortAudio is missing. On Ubuntu:\n"
            "    sudo apt install libportaudio2"
        ) from e
    except ImportError as e:
        raise RuntimeError("no sounddevice. Install it with: uv sync") from e
    return sd


def _import_faster_whisper():
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError("no faster-whisper. Install it with: uv sync") from e
    return WhisperModel
