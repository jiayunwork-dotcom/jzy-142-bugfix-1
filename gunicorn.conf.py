"""gunicorn 运行配置。

刻意使用单 worker、多线程：
- 作业由**进程内**后台调度线程串行执行，多 worker 会各自启动调度线程，
  对同一个 SQLite 队列造成重复领取（SQLite 只能提供单进程串行写入）；
- 反应谱计算的热点在 NumPy（底层 BLAS 已并行），多线程仅承担 HTTP
  请求的 I/O 等待。

如需横向扩展，应把作业队列换成外部消息/任务系统，而不是增加 worker。
"""

import os

bind = os.environ.get("BIND", "0.0.0.0:8000")
workers = 1
threads = int(os.environ.get("GUNICORN_THREADS", "8"))
worker_class = "gthread"
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
graceful_timeout = 30
keepalive = 5

# 计算型请求可能上传较大批次，给足预读时间与缓冲
limit_request_line = 0
limit_request_field_size = 0

# 预加载：确保应用工厂（含调度线程）只在 master 之后的唯一 worker 内启动一次
preload_app = False
