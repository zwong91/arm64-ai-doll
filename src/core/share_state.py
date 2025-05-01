import threading
from typing import Callable, Optional

class State:
    _listen = True
    _lock = threading.Lock()
    _on_change: Optional[Callable[[bool], None]] = None

    @classmethod
    def listening(cls) -> bool:
        with cls._lock:
            return cls._listen

    @classmethod
    def pause_listening(cls):
        cls._set_listen(False)

    @classmethod
    def resume_listening(cls):
        cls._set_listen(True)

    @classmethod
    def _set_listen(cls, value: bool):
        with cls._lock:
            if cls._listen != value:
                cls._listen = value
                if cls._on_change:
                    cls._on_change(value)

    @classmethod
    def set_on_change_callback(cls, callback: Callable[[bool], None]):
        """设置状态变更时触发的回调函数，参数为当前状态（True/False）"""
        with cls._lock:
            cls._on_change = callback
