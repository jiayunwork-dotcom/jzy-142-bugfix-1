FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SEISMIC_DB=/data/seismic.db

WORKDIR /app

# 先装依赖（利用层缓存）
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 再拷贝应用代码
COPY app ./app
COPY wsgi.py gunicorn.conf.py ./

# 回归测试随镜像分发（requirements 已含 pytest），
# 容器内可直接 `pytest -q`（如 docker compose exec seismic-spectrum pytest -q）
COPY tests ./tests
COPY pytest.ini ./

# SQLite 文件所在挂载卷
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# 容器停止时给调度线程留出收尾窗口（中断作业会在重启后标记 interrupted）
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["gunicorn", "-c", "gunicorn.conf.py", "wsgi:app"]
