# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('sherpa/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01', 'sherpa/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01'), ('sherpa/sherpa-onnx-streaming-paraformer-bilingual-zh-en', 'sherpa/sherpa-onnx-streaming-paraformer-bilingual-zh-en'), ('sensevoice_ckpt', 'sensevoice_ckpt'), ('whisper_ckpt', 'whisper_ckpt'), ('vad_ckpt', 'vad_ckpt'), ('sherpa/vits-icefall-zh-aishell3', 'sherpa/vits-icefall-zh-aishell3'), ('speech-enhancement', 'speech-enhancement'), ('MiniMind2-Small', 'MiniMind2-Small'), ('model/minimind_tokenizer', 'model/minimind_tokenizer'), ('keywords', 'keywords')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='arm64_ai_doll',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='arm64_ai_doll',
)
