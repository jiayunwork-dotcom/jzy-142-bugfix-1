"""Flask 应用工厂与后台调度器装配。"""

from __future__ import annotations

import os

from flask import Flask, jsonify

from .errors import SeismicError
from .scheduler import Scheduler
from .storage import Storage


def create_app(db_path: str | None = None, *, start_worker: bool = True) -> Flask:
    app = Flask(__name__)
    db_path = db_path or os.environ.get("SEISMIC_DB", "/data/seismic.db")
    storage = Storage(db_path)
    scheduler = Scheduler(storage)

    app.extensions["storage"] = storage
    app.extensions["scheduler"] = scheduler

    from .api.records import bp as records_bp
    from .api.jobs import bp as jobs_bp

    app.register_blueprint(records_bp)
    app.register_blueprint(jobs_bp)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "db": db_path})

    @app.errorhandler(SeismicError)
    def _seismic_error(exc):  # pragma: no cover - 领域错误已在接口内转换
        return jsonify({"error": str(exc)}), 400

    @app.errorhandler(404)
    def _not_found(exc):  # noqa: ANN001
        return jsonify({"error": "接口不存在"}), 404

    @app.errorhandler(405)
    def _method_not_allowed(exc):  # noqa: ANN001
        return jsonify({"error": "HTTP 方法不允许"}), 405

    if start_worker:
        scheduler.start()

    return app
