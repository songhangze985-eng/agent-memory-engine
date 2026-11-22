"""认知记忆引擎（v2 VFS 虚拟文件系统主引擎 + v1 账本兼容入口）。"""

from memengine.engine import CognitiveMemoryEngine
from memengine.vfs import OptimisticLockError, VFSMemoryEngine, open_vfs_store

__all__ = ["VFSMemoryEngine", "open_vfs_store", "OptimisticLockError", "CognitiveMemoryEngine"]
__version__ = "0.2.0"
