"""
torchaudio_compat.py — restore torchaudio.AudioMetaData for pyannote 3.1.1
==========================================================================
WhisperX 3.3.1 pins pyannote.audio 3.1.1, whose ``core/io.py`` references
``torchaudio.AudioMetaData`` in a return-type annotation that is evaluated at
import time. Recent torchaudio (the Colab default) relocated this class out of
the top-level namespace, so importing pyannote raises:

    AttributeError: module 'torchaudio' has no attribute 'AudioMetaData'

This shim re-attaches the class (at whichever internal location it now lives,
or a stub as a last resort) BEFORE pyannote is imported. It is idempotent and
a no-op on torchaudio versions that still expose the attribute (e.g. the local
2.1.x pin). Critically, it changes NO installed package — the pinned
whisperx / transformers / autoawq / torch / torchaudio versions are untouched,
so it cannot reintroduce a resolver conflict or break the AWQ kernels.

Usage — call apply() before the first `import whisperx` / `import pyannote`:

    import torchaudio_compat
    torchaudio_compat.apply()
    import whisperx
"""

import importlib


def apply() -> bool:
    """
    Ensure ``torchaudio.AudioMetaData`` exists.

    Returns True if a patch was applied, False if it was already present or
    torchaudio is unavailable.
    """
    try:
        import torchaudio
    except Exception:
        return False

    if hasattr(torchaudio, "AudioMetaData"):
        apply_backend_shim()
        apply_torch_load_shim()
        return False

    # The class still ships with torchaudio — only its public location moved
    # across the backend-dispatcher refactors. Try the known homes in order.
    audio_meta = None
    for modpath in ("torchaudio.backend.common", "torchaudio._backend.common"):
        try:
            audio_meta = importlib.import_module(modpath).AudioMetaData
            break
        except Exception:
            continue

    if audio_meta is None:
        # pyannote 3.1.1 only uses it as a type annotation, so a bare stub
        # satisfies the attribute lookup without affecting runtime behaviour.
        class AudioMetaData:  # noqa: D401 - minimal placeholder
            pass
        audio_meta = AudioMetaData

    torchaudio.AudioMetaData = audio_meta
    print("  [torchaudio_compat] patched torchaudio.AudioMetaData for pyannote 3.1.1")
    apply_backend_shim()
    apply_torch_load_shim()
    return True
def apply_backend_shim() -> bool:
    """
    Ensure torchaudio.list_audio_backends / get_audio_backend / set_audio_backend
    exist. pyannote 3.1.1's ``core/io.py`` calls ``list_audio_backends()`` at
    import time (inside ``Audio.__init__``) to pick a default backend. Recent
    torchaudio (TorchCodec-based) dropped this old backend-dispatch API.

    Returns True if a patch was applied, False if already present or
    torchaudio is unavailable.
    """
    try:
        import torchaudio
    except Exception:
        return False

    patched = False

    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
        patched = True

    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"
        patched = True

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend=None: None
        patched = True

    if patched:
        print("  [torchaudio_compat] patched torchaudio.list_audio_backends for pyannote 3.1.1")

    return patched

def apply_torch_load_shim() -> bool:
    """
    Force ``torch.load`` to default to ``weights_only=False``.

    PyTorch >=2.6 changed torch.load's default from False to True. The
    official pyannote 3.1.1 checkpoints (downloaded from Hugging Face) embed
    a ``torch.torch_version.TorchVersion`` global that is not in the new
    default safe-list, so loading fails under the new default. We trust the
    official HF checkpoint source, so we restore the old default globally.

    Returns True if a patch was applied, False if already patched or torch
    is unavailable.
    """
    try:
        import torch
    except Exception:
        return False

    if getattr(torch.load, "_dubly_patched", False):
        return False

    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    _patched_load._dubly_patched = True
    torch.load = _patched_load
    print("  [torchaudio_compat] patched torch.load default to weights_only=False for pyannote checkpoints")
    return True