from __future__ import annotations

import io
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import imageio_ffmpeg
import numpy as np
import pandas as pd
import plotly.express as px
import soundfile as sf
import streamlit as st


PITCH_NAMES = ["C", "C♯/D♭", "D", "D♯/E♭", "E", "F", "F♯/G♭", "G", "G♯/A♭", "A", "A♯/B♭", "B"]
SWARA_NAMES = ["Sa", "Ri₁", "Ri₂ / Ga₁", "Ri₃ / Ga₂", "Ga₃", "Ma₁", "Ma₂", "Pa", "Da₁", "Da₂ / Ni₁", "Da₃ / Ni₂", "Ni₃"]

# A small reference catalog for an explicitly experimental scale-match display.
# These are swara positions relative to Sa, represented in semitones.
RAGA_SCALE_CATALOG = {
    "Mohanam": {0, 2, 4, 7, 9},
    "Hamsadhwani": {0, 2, 4, 7, 11},
    "Hindolam": {0, 3, 5, 8, 10},
    "Shankarabharanam": {0, 2, 4, 5, 7, 9, 11},
    "Kalyani": {0, 2, 4, 6, 7, 9, 11},
    "Kharaharapriya": {0, 2, 3, 5, 7, 9, 10},
    "Mayamalavagowla": {0, 1, 4, 5, 7, 8, 11},
    "Charukesi": {0, 2, 4, 5, 7, 8, 10},
    "Keeravani": {0, 2, 3, 5, 7, 8, 11},
    "Abheri": {0, 2, 3, 5, 7, 9, 10},
}


@dataclass
class Analysis:
    sr: int
    duration: float
    pitches: np.ndarray
    times: np.ndarray
    chroma: np.ndarray
    tonic_midi: int | None
    tonic_confidence: float


def decode_audio(data: bytes, filename: str) -> tuple[np.ndarray, int, bytes]:
    """Decode an audio or video file and return mono audio, rate, and playable audio bytes."""
    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
    except Exception:
        # soundfile cannot decode many compressed formats or video containers.
        # Use the FFmpeg binary bundled by imageio-ffmpeg to extract the first audio stream.
        suffix = Path(filename).suffix
        if len(suffix) > 12:
            suffix = ".bin"
        try:
            with tempfile.TemporaryDirectory(prefix="raga_audio_") as temp_dir:
                input_path = Path(temp_dir) / f"input{suffix}"
                output_path = Path(temp_dir) / "audio.wav"
                input_path.write_bytes(data)
                subprocess.run(
                    [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-hide_banner", "-loglevel", "error",
                     "-i", str(input_path), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "22050",
                     "-c:a", "pcm_s16le", str(output_path)],
                    check=True, capture_output=True, timeout=180,
                )
                y, sr = sf.read(str(output_path), dtype="float32", always_2d=True)
                y = y.mean(axis=1)
        except Exception as exc:
            raise ValueError(
                "I couldn't find a readable audio track in this file. Try an audio file or a video "
                "with sound (such as MP4, MOV, or WebM)."
            ) from exc
    if y.size == 0:
        raise ValueError("The uploaded audio is empty or could not be decoded.")
    if len(y) > sr * 60 * 20:
        raise ValueError("For this first version, please upload a song shorter than 20 minutes.")
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, y, sr, format="WAV", subtype="PCM_16")
    playable_audio = wav_buffer.getvalue()
    return y.astype(np.float32), int(sr), playable_audio


def analyze(y: np.ndarray, sr: int) -> Analysis:
    # Reduce large uploads to a manageable analysis rate while retaining musical pitch.
    target_sr = 22050
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    hop = 512
    f0, voiced, probabilities = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
        sr=sr, frame_length=2048, hop_length=hop,
    )
    midi = librosa.hz_to_midi(f0)
    valid = np.isfinite(midi) & voiced & (probabilities > 0.35)
    pitches = midi[valid]
    times = librosa.times_like(f0, sr=sr, hop_length=hop)[valid]
    if pitches.size:
        pcs = np.mod(np.rint(pitches).astype(int), 12)
        # Weight stable voiced frames by pyin confidence; use pitch-class prominence as a tonic hint.
        weights = np.nan_to_num(probabilities[valid], nan=0.0)
        histogram = np.bincount(pcs, weights=weights, minlength=12).astype(float)
        smoothed = histogram + np.roll(histogram, 1) * 0.25 + np.roll(histogram, -1) * 0.25
        order = np.argsort(smoothed)[::-1]
        tonic_pc = int(order[0])
        confidence = float(smoothed[order[0]] / max(smoothed.sum(), 1e-9))
        tonic_midi = int(round(np.median(pitches)))
        tonic_midi += (tonic_pc - tonic_midi) % 12
        # Put the tonic near the central register of detected pitches.
        tonic_midi += 12 * round((float(np.median(pitches)) - tonic_midi) / 12)
    else:
        histogram = np.zeros(12)
        tonic_midi = None
        confidence = 0.0
    return Analysis(sr, len(y) / sr, pitches, times, histogram, tonic_midi, confidence)


def display_note(midi: int) -> str:
    return f"{PITCH_NAMES[midi % 12]}{midi // 12 - 1}"


def scale_matches(pitch_counts: np.ndarray, tonic_pc: int) -> list[tuple[str, float]]:
    """Rank catalog scales by how much of the observed pitch mass they explain.

    This is not a raga classifier: pitch-class membership omits characteristic
    phrases, gamakas, ascent/descent rules, and other raga-specific behavior.
    """
    total = float(pitch_counts.sum())
    if total <= 0:
        return []
    ranked = []
    for name, swaras in RAGA_SCALE_CATALOG.items():
        pitch_classes = {(tonic_pc + interval) % 12 for interval in swaras}
        explained = sum(float(pitch_counts[pc]) for pc in pitch_classes) / total
        observed_coverage = sum(1 for pc in pitch_classes if pitch_counts[pc] > 0) / len(swaras)
        ranked.append((name, 100 * (0.8 * explained + 0.2 * observed_coverage)))
    return sorted(ranked, key=lambda item: item[1], reverse=True)


st.set_page_config(page_title="Raga & Pitch Explorer", page_icon="🎼", layout="wide")
st.title("🎼 Raga & Pitch Explorer")
st.write("Upload a song to inspect its pitch, likely tonic, observed notes, and transposition interval.")
st.info("This version estimates pitch and tonic, and shows tentative matches to a small raga scale catalog. Those matches are exploratory, not reliable raga identification; a validated model needs representative labeled recordings and analysis of characteristic melodic phrases.")

with st.sidebar:
    st.header("What this can tell you")
    st.markdown("- **Tonic hint:** most prominent pitch class, which may be a drone or recurring note. Confirm it by ear.\n- **Pitch notes:** approximate equal-tempered note names from the melody.\n- **Swara positions:** semitone positions relative to the selected tonic; several Carnatic swara names share enharmonic pitches.\n- **Transpose:** semitone shift to move the detected tonic to a chosen target.")
    st.caption("Polyphonic music, percussion, changing keys, and strong accompaniment can confuse pitch tracking.")

upload = st.file_uploader("Choose an audio or video file", type=None, help="Audio files and video files with a sound track are supported. Files without audio cannot be analyzed.")
if upload:
    try:
        audio, sample_rate, playable_audio = decode_audio(upload.getvalue(), upload.name)
        with st.spinner("Listening for stable pitched notes…"):
            result = analyze(audio, sample_rate)
        st.audio(playable_audio, format="audio/wav")
        st.caption(f"Duration: {result.duration:.1f} seconds · Analysis sample rate: {result.sr:,} Hz")
        if result.tonic_midi is None:
            st.warning("No stable melodic pitches were detected. Try a clearer recording or a section with a sustained melody.")
            st.stop()

        tonic = result.tonic_midi
        tonic_pc = tonic % 12
        confidence_label = "low" if result.tonic_confidence < 0.13 else ("moderate" if result.tonic_confidence < 0.2 else "strong")
        a, b, c = st.columns(3)
        a.metric("Tonic hint", display_note(tonic), f"{confidence_label} pitch-class prominence")
        b.metric("Detected pitched frames", f"{len(result.pitches):,}")
        c.metric("Raga model", "Not configured", "Scale matches below are exploratory")
        st.caption(f"The tonic hint is an automatic estimate (prominence {result.tonic_confidence:.0%}), not a verified Sa. Set the tonic manually if you know it.")

        with st.expander("Set tonic manually", expanded=False):
            tonic_choice = st.selectbox("Known Sa", PITCH_NAMES, index=tonic_pc)
            tonic_pc = PITCH_NAMES.index(tonic_choice)
        observed = np.mod(np.rint(result.pitches).astype(int), 12)
        counts = np.bincount(observed, minlength=12)
        # Pitch class includes octave variants; rows show all chromatic offsets from chosen Sa.
        rows = []
        for offset in range(12):
            pc = (tonic_pc + offset) % 12
            rows.append({"Offset from Sa": offset, "Pitch class": PITCH_NAMES[pc], "Carnatic position (approx.)": SWARA_NAMES[offset], "Observed frames": int(counts[pc])})
        table = pd.DataFrame(rows)
        matches = scale_matches(counts, tonic_pc)
        st.subheader("Experimental raga scale matches")
        st.caption("These are the closest matches in a small reference scale list, not reliable raga identification. Ragam identity also depends on characteristic phrases and gamakas, which this baseline does not analyze.")
        if matches:
            match_df = pd.DataFrame(matches[:3], columns=["Raga scale", "Scale resemblance (heuristic)"])
            match_df["Scale resemblance (heuristic)"] = match_df["Scale resemblance (heuristic)"].map(lambda score: f"{score:.0f}%")
            st.dataframe(match_df, hide_index=True, use_container_width=True)
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Observed pitch classes")
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.caption("Pitch names and swara positions are approximate: tuning, gamakas, and pitch bends are not represented by fixed semitone labels.")
        with right:
            st.subheader("Pitch-class prominence")
            plot_data = pd.DataFrame({"Pitch class": PITCH_NAMES, "Weighted voiced frames": result.chroma})
            fig = px.bar(plot_data, x="Pitch class", y="Weighted voiced frames", color="Pitch class")
            fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Transpose guidance")
        target = st.selectbox("Target Sa", PITCH_NAMES, index=tonic_pc)
        delta = (PITCH_NAMES.index(target) - tonic_pc + 6) % 12 - 6
        if delta == 0:
            st.success(f"No change: keep Sa at {target}.")
        else:
            direction = "up" if delta > 0 else "down"
            st.success(f"Shift the audio {direction} by {abs(delta)} semitone(s) to move the estimated tonic to {target}.")
        st.caption("This describes a pitch shift. The app does not modify or export transposed audio.")
    except Exception as exc:
        st.error(f"Could not analyze this audio: {exc}")
else:
    st.markdown("Upload an audio file or a video with sound, including formats such as MP4, MOV, and WebM. A file must contain an audio track; images and documents cannot be analyzed as songs. Start with a short, clear recording for the most useful pitch estimate.")

st.divider()
st.caption("The scale catalog is a starting point for future model development. Add and validate richer raga features before treating a candidate as an identification.")
