# -*- mode: python ; coding: utf-8 -*-
datas = []
datas += [("/opt/anaconda3/envs/gm_py39/lib/python3.9/site-packages/onnxruntime/capi/libonnxruntime_providers_shared.so", "onnxruntime/capi")]

a = Analysis(
    ['service_run.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['gunicorn.glogging'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='service_run',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
