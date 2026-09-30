"""gunicorn 入口：``gunicorn wsgi:app``。

SQLite 文件通过环境变量 ``SEISMIC_DB`` 指向挂载卷（见 gunicorn.conf.py）。
"""

from app import create_app

app = create_app()
