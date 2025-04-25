import os
import sys
import ctypes

def load_portaudio():
    """在程序启动时加载 PortAudio 库"""
    try:
        # Linux
        if sys.platform == 'linux':
            lib_path = os.path.join(os._MEIPASS, 'libportaudio.so.2')
            if os.path.exists(lib_path):
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            else:
                ctypes.CDLL('libportaudio.so.2', mode=ctypes.RTLD_GLOBAL)
        # macOS 
        elif sys.platform == 'darwin':
            lib_path = os.path.join(os._MEIPASS, 'libportaudio.2.dylib')
            if os.path.exists(lib_path):
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            else:
                ctypes.CDLL('libportaudio.2.dylib', mode=ctypes.RTLD_GLOBAL)
    except Exception as e:
        print(f"Failed to load PortAudio: {e}")

# 程序启动时自动执行
load_portaudio()