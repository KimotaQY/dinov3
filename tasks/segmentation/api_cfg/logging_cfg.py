import os

# 确保日志目录存在
log_directory = "/tmp/dino_segmentation"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# 日志配置
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(asctime)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt":
            '%(levelprefix)s %(asctime)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": f"{log_directory}/api.log",
            "maxBytes": 1024 * 1024 * 10,  # 10 MB
            "backupCount": 5,
            "formatter": "default",
        },
        "task_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": f"{log_directory}/tasks.log",
            "maxBytes": 1024 * 1024 * 10,  # 10 MB
            "backupCount": 5,
            "formatter": "default",
        }
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO"
        },
        "uvicorn.error": {
            "level": "INFO"
        },
        "uvicorn.access": {
            "handlers": ["access"],
            "level": "INFO",
            "propagate": False
        },
        "dino_segmentation_api": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": False
        },
        "dino_segmentation_tasks": {
            "handlers": ["task_file"],
            "level": "INFO",
            "propagate": False
        },
    },
}
