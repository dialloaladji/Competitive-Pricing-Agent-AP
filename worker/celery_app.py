from celery import Celery
from celery.schedules import crontab
from api.config import settings

celery_app = Celery(
    "pricing_agent",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

celery_app.conf.beat_schedule = {
    "track-prices-every-hour": {
        "task": "track_all_prices",
        "schedule": crontab(minute=f"*/{settings.price_track_interval_minutes}"),
    },
    "check-alerts": {
        "task": "check_price_alerts",
        "schedule": crontab(minute=f"*/{settings.alert_check_interval_minutes}"),
    },
}

celery_app.autodiscover_tasks(["worker.tasks"])
