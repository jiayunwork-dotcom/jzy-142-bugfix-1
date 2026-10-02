FROM python:3.12-slim AS base

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

# 测试镜像：在运行镜像之上加 pytest、测试代码与开发依赖。
# 用法：
#   docker build --target test -t seismic-spectrum:test .
#   docker run --rm seismic-spectrum:test            # 跑全部 pytest
# 默认构建目标（不带 --target）仍是下面的精简运行镜像。
FROM base AS test
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY tests ./tests
COPY pytest.ini ./
CMD ["pytest", "-q"]

FROM base AS runtime

# SQLite 文件所在挂载卷
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# 容器停止时给调度线程留出收尾窗口（取消/中断作业在重启后恢复）
STOPSIGNAL SIGTERM
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["gunicorn", "-c", "gunicorn.conf.py", "wsgi:app"]
