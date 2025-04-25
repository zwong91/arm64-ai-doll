import os
import sys
import ctypes

def load_portaudio():
    """加载 PortAudio 库"""
    try:
        # 获取库路径 - 优先使用打包后路径
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        
        if sys.platform == 'linux':
            lib_names = ['libportaudio.so.2', 'libportaudio.so']
            for name in lib_names:
                try:
                    lib_path = os.path.join(base_path, name)
                    if os.path.exists(lib_path):
                        print(f"Loading PortAudio from: {lib_path}")
                        return ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                    else:
                        print(f"Loading system PortAudio: {name}")
                        return ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
                except Exception as e:
                    print(f"Failed to load {name}: {e}")
                    continue
                    
        elif sys.platform == 'darwin':
            lib_names = ['libportaudio.2.dylib', 'libportaudio.dylib'] 
            for name in lib_names:
                try:
                    lib_path = os.path.join(base_path, name)
                    if os.path.exists(lib_path):
                        print(f"Loading PortAudio from: {lib_path}")
                        return ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                    else:
                        print(f"Loading system PortAudio: {name}")
                        return ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
                except Exception as e:
                    print(f"Failed to load {name}: {e}")
                    continue
                    
        raise RuntimeError("无法加载 PortAudio 库")
        
    except Exception as e:
        print(f"加载 PortAudio 失败: {e}")
        raise

load_portaudio()