import os
import sys
import ctypes

def load_portaudio():
    try:
        # 获取可执行文件目录
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))

        # 尝试加载打包的库文件
        lib_path = os.path.join(base_path, 'libportaudio.so.2')
        if os.path.exists(lib_path):
            print(f"正在加载 PortAudio: {lib_path}")
            return ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        
        # 回退到系统库
        print("尝试加载系统 PortAudio")
        return ctypes.CDLL('libportaudio.so.2', mode=ctypes.RTLD_GLOBAL)
    
    except Exception as e:
        print(f"加载 PortAudio 失败: {e}")
        raise

# 程序启动时执行
load_portaudio()