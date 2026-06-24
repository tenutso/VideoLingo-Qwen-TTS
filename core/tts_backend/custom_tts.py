import threading
from pathlib import Path
import torch
import soundfile as sf
from core.utils import load_key
from core.utils.models import _AUDIO_REFERS_DIR

# Qwen3-TTS (https://github.com/QwenLM/Qwen3-TTS) is a local Hugging Face model.
# Three modes are supported via config `qwen_tts.mode`:
#   custom  -> CustomVoice model, a preset speaker (+ optional `instruct` style)
#   design  -> VoiceDesign model, a voice described in natural language (`instruct`)
#   clone   -> Base model, clones the original speaker from reference audio that
#              VideoLingo already extracts to output/audio/refers/<number>.wav
#
# gen_audio runs TTS with a ThreadPoolExecutor (max_workers), so the shared model,
# the cached clone prompt, and the GPU generate calls are all guarded by _LOCK.
_MODEL = None
_MODEL_ID = None
_CLONE_PROMPT = None
_LOCK = threading.Lock()
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

def _get_model(model_id):
    """Lazily load (and cache) the Qwen3-TTS model for the given id. Thread-safe."""
    global _MODEL, _MODEL_ID
    if _MODEL is not None and _MODEL_ID == model_id:
        return _MODEL
    with _LOCK:
        if _MODEL is None or _MODEL_ID != model_id:
            from qwen_tts import Qwen3TTSModel
            cfg = load_key("qwen_tts")
            _MODEL = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=cfg.get("device", "cuda:0"),
                dtype=_DTYPES.get(cfg.get("dtype", "bfloat16"), torch.bfloat16),
            )
            _MODEL_ID = model_id
    return _MODEL

def _lang_kwargs(cfg):
    """'Auto' (or empty) lets the model auto-detect the language."""
    lang = cfg.get("language")
    return {} if (not lang or lang == "Auto") else {"language": lang}

def _get_clone_prompt(model, task_df):
    """Build (once) a reusable voice-clone prompt from a combined ~10s reference of
    the original speaker. Returns None if no usable reference could be assembled."""
    global _CLONE_PROMPT
    if _CLONE_PROMPT is not None:
        return _CLONE_PROMPT
    with _LOCK:
        if _CLONE_PROMPT is None:
            # Reuse VideoLingo's existing combined-reference builder (imported lazily
            # to avoid any import-order coupling with the TTS dispatcher).
            from core.tts_backend.sf_fishtts import get_ref_audio
            ref_audio, ref_text = get_ref_audio(task_df)
            if ref_audio is None or ref_text is None:
                return None
            _CLONE_PROMPT = model.create_voice_clone_prompt(
                ref_audio=str(ref_audio), ref_text=ref_text
            )
    return _CLONE_PROMPT

def _generate_clone(text, cfg, lang_kw, number, task_df):
    model = _get_model(cfg.get("model_clone", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"))
    ref_mode = cfg.get("ref_mode", "combined")

    if ref_mode == "combined":
        prompt = _get_clone_prompt(model, task_df)
        if prompt is not None:
            with _LOCK:
                return model.generate_voice_clone(
                    text=text, voice_clone_prompt=prompt, **lang_kw
                )
        # combined reference unavailable -> fall back to per-line below

    # per_line: clone from this line's own reference clip + its original transcript
    if number is None or task_df is None:
        raise ValueError("clone mode requires `number` and `task_df`")
    ref_audio = f"{_AUDIO_REFERS_DIR}/{number}.wav"
    if not Path(ref_audio).exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
    ref_text = task_df[task_df["number"] == number]["origin"].iloc[0]
    with _LOCK:
        return model.generate_voice_clone(
            text=text, ref_audio=ref_audio, ref_text=ref_text, **lang_kw
        )

def custom_tts(text, save_path, number=None, task_df=None):
    """
    Custom TTS interface backed by Qwen3-TTS (local model).

    Args:
        text (str): Text to be converted to speech
        save_path (str): Path to save the audio file (WAV)
        number (int, optional): Subtitle line number (required for clone mode)
        task_df (DataFrame, optional): Task table with 'number'/'origin'/'duration'
            columns (required for clone mode)

    Returns:
        None
    """
    cfg = load_key("qwen_tts")
    mode = cfg.get("mode", "custom")
    lang_kw = _lang_kwargs(cfg)

    speech_file_path = Path(save_path)
    speech_file_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "clone":
        wavs, sr = _generate_clone(text, cfg, lang_kw, number, task_df)
    elif mode == "design":
        model = _get_model(cfg.get("model_design", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"))
        with _LOCK:
            wavs, sr = model.generate_voice_design(
                text=text, instruct=cfg.get("instruct", ""), **lang_kw
            )
    else:  # custom
        model = _get_model(cfg.get("model_custom", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
        kwargs = {"text": text, "speaker": cfg.get("speaker", "Ryan"), **lang_kw}
        if cfg.get("instruct"):
            kwargs["instruct"] = cfg["instruct"]
        with _LOCK:
            wavs, sr = model.generate_custom_voice(**kwargs)

    sf.write(str(speech_file_path), wavs[0], sr)
    print(f"Audio saved to {speech_file_path}")

if __name__ == "__main__":
    # Test example (custom mode; clone mode needs reference audio + task_df)
    custom_tts("This is a test.", "custom_tts_test.wav")
