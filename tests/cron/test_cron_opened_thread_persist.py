"""Persisting an opened delivery thread back to ``origin.thread_id``.

Once a thread-preferred target (e.g. a Discord channel with threads enabled)
receives its first delivery, a NEW thread is opened per fire unless the opened
id is stamped back into the job's ``origin`` — every re-run would otherwise
spawn a duplicate thread and seed a fresh session each time. The persist must
only happen when the delivered target IS the origin conversation (same platform
+ chat), must respect the Slack thread-per-message artifact heuristic, must be
idempotent (same id → no write), and must never fail a completed delivery.
"""

import pytest

from cron import scheduler_delivery as sched_delivery


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with temp HERMES_HOME."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    return hermes_home


class _TargetDeliveryStub:
    """Minimal _TargetDelivery: only the fields _seed_live_delivery_sessions reads."""

    def __init__(self, job, origin, opened_thread_id, platform_name="discord", chat_id="9001"):
        self.job = job
        self.origin = origin
        self.opened_thread_id = opened_thread_id
        self.platform_name = platform_name
        self.chat_id = chat_id
        self.runtime_adapter = None
        self.mirror_text = "brief"
        self.is_dm_target = False
        self.in_channel_surface = False
        self.inchannel_continuable = False
        self.mirror_this_target = False
        self.thread_id = None
        self.origin_user_id = None


def _job_record(env, job_id="job-t1", origin=None):
    import cron.jobs as jobs_mod
    record = {
        "id": job_id,
        "name": job_id,
        "schedule": "1h",
        "enabled": False,
        "prompt": "noop",
        "origin": dict(origin or {}),
    }
    jobs_mod.save_jobs([record])
    return record


def _job_origin(env, job_id):
    import cron.jobs as jobs_mod
    jobs = jobs_mod.load_jobs()
    return next(j for j in jobs if j["id"] == job_id)["origin"]


def test_opened_thread_persisted_to_origin(cron_env, monkeypatch):
    """First delivery to the origin conversation stamps the opened thread into origin."""
    import cron.jobs as jobs_mod
    origin = {"platform": "discord", "chat_id": "9001", "thread_id": None}
    record = _job_record(cron_env, origin=origin)

    monkeypatch.setattr(
        sched_delivery, "_seed_cron_thread_session", lambda *a, **k: True)
    monkeypatch.setattr(
        sched_delivery, "_maybe_mirror_cron_delivery", lambda *a, **k: None)

    t = _TargetDeliveryStub(record, origin, opened_thread_id="777")
    sched_delivery._seed_live_delivery_sessions(t, delivered_message_id=None)

    updated = _job_origin(cron_env, "job-t1")
    assert updated["thread_id"] == "777"
    # The record the caller holds is not the store's copy — the persist re-reads
    # the stored origin, so a stale in-memory job dict must not leak back.
    assert record["origin"]["thread_id"] is None


def test_other_target_never_rewrites_origin(cron_env, monkeypatch):
    """A thread opened for a NON-origin target must not rewrite deliver=origin routing."""
    origin = {"platform": "discord", "chat_id": "4242", "thread_id": None}
    record = _job_record(cron_env, origin=origin)

    monkeypatch.setattr(
        sched_delivery, "_seed_cron_thread_session", lambda *a, **k: True)
    monkeypatch.setattr(
        sched_delivery, "_maybe_mirror_cron_delivery", lambda *a, **k: None)

    t = _TargetDeliveryStub(record, origin, opened_thread_id="777", chat_id="9001")
    sched_delivery._seed_live_delivery_sessions(t, delivered_message_id=None)

    assert _job_origin(cron_env, "job-t1").get("thread_id") is None


def test_idempotent_no_rewrite_when_same_thread(cron_env, monkeypatch):
    """Already-persisted id → no second write (and no exception)."""
    origin = {"platform": "discord", "chat_id": "9001", "thread_id": "777"}
    record = _job_record(cron_env, origin=origin)

    monkeypatch.setattr(
        sched_delivery, "_seed_cron_thread_session", lambda *a, **k: True)
    monkeypatch.setattr(
        sched_delivery, "_maybe_mirror_cron_delivery", lambda *a, **k: None)

    def _fail_update(job_id, updates):
        raise AssertionError("update_job must not be called when thread_id already matches")

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "update_job", _fail_update)

    t = _TargetDeliveryStub(record, origin, opened_thread_id="777")
    sched_delivery._seed_live_delivery_sessions(t, delivered_message_id=None)


def test_slack_home_artifact_thread_not_persisted(cron_env, monkeypatch):
    """Slack thread-per-message artifact (origin chat == home chat) must not be pinned —
    _origin_thread_is_stale would drop it next run and reopen anyway."""
    origin = {"platform": "slack", "chat_id": "C123", "thread_id": None}
    record = _job_record(cron_env, origin=origin)

    monkeypatch.setattr(
        sched_delivery, "_seed_cron_thread_session", lambda *a, **k: True)
    monkeypatch.setattr(
        sched_delivery, "_maybe_mirror_cron_delivery", lambda *a, **k: None)
    monkeypatch.setattr(sched_delivery, "_get_home_target_chat_id", lambda platform: "C123")

    t = _TargetDeliveryStub(record, origin, opened_thread_id="777",
                            platform_name="slack", chat_id="C123")
    sched_delivery._seed_live_delivery_sessions(t, delivered_message_id=None)

    assert _job_origin(cron_env, "job-t1").get("thread_id") is None


def test_persist_failure_never_fails_delivery(cron_env, monkeypatch):
    """update_job raising must not propagate — bookkeeping never fails a completed delivery."""
    origin = {"platform": "discord", "chat_id": "9001", "thread_id": None}
    record = _job_record(cron_env, origin=origin)

    monkeypatch.setattr(
        sched_delivery, "_seed_cron_thread_session", lambda *a, **k: True)
    monkeypatch.setattr(
        sched_delivery, "_maybe_mirror_cron_delivery", lambda *a, **k: None)

    def _boom(job_id, updates):
        raise RuntimeError("store unavailable")

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "update_job", _boom)

    t = _TargetDeliveryStub(record, origin, opened_thread_id="777")
    # Must not raise.
    sched_delivery._seed_live_delivery_sessions(t, delivered_message_id=None)
