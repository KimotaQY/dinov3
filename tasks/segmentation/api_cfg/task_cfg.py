from enum import IntEnum
from typing import Dict, Any


class TaskStatus(IntEnum):
    NOT_STARTED = 0
    PROCESSING = 1
    COMPLETED = 2
    FAILED = 3


# 队列配置
QUEUE_CONFIG = {
    "queue_name": "dino_segment",
    "queue_size": 100,
    "queue_timeout": 60,
    "queue_retry": 3,
    "queue_retry_delay": 5,
    "queue_retry_max": 10,
    "queue_retry_max_delay": 60,
    "queue_retry_max_delay_multiplier": 2,
}
QUEUE = []
# 任务存储结构
TASK_STORE: Dict[str, Dict[str, Any]] = {}

# 队列持久化文件路径
QUEUE_PERSISTENCE_FILE = "/tmp/dino_segmentation/queue.pkl"
TASK_STORE_PERSISTENCE_FILE = "/tmp/dino_segmentation/task_store.pkl"
