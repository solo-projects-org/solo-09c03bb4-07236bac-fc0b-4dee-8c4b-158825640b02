"""优化前后等价性：同一清册、同一批次序列，旧实现与新实现的回执/名单必须一致。

旧实现（逐票 SQL）原样保存在 tests/legacy_baseline/，本测试在两个独立数据库上
分别驱动两套 service，断言：
- 每条回执除 created_at 外逐字段相等（seq、unknown_invoices、removed、补样结果）；
- 最终现行名单完全相等；
- 同轮同批号重送同组票返回原回执、换组报 409 的行为两边一致。
"""
import json
import os
import random

import pytest

import app.db as db
from app import service as new_service
from tests.legacy_baseline import legacy_service


class _Topic:
    def __init__(self, topic_code, quota, samples, candidates):
        self.topic_code = topic_code
        self.quota = quota
        self.samples = samples
        self.candidates = candidates

    def model_dump(self):
        return {
            "topic_code": self.topic_code,
            "quota": self.quota,
            "samples": self.samples,
            "candidates": self.candidates,
        }


class _Payload:
    def __init__(self, round_id, topics):
        self.round_id = round_id
        self.topics = topics

    def model_dump(self):
        return {"round_id": self.round_id, "topics": [t.model_dump() for t in self.topics]}


def _make_scenario(seed=20261120):
    """构造含共享备选、跨专题重叠、票号形态各异的确定性场景。"""
    rng = random.Random(seed)
    topics = []
    pool = [f"C{idx:04d}" for idx in range(600)]  # 共享备选池
    sample_idx = 0
    for i in range(8):
        quota = rng.randint(8, 20)
        samples = [f"S{i}-{j:03d}" for j in range(quota)]
        sample_idx += quota
        k = rng.randint(20, 80)
        candidates = rng.sample(pool, k)  # 同一备选可被多个专题抽中
        topics.append(_Topic(f"T{i}", quota, samples, candidates))
    return _Payload("REQ", topics)


def _make_batches(payload):
    """固定批次序列：大中小批、重复票、未知票、已退出票重送、共享备选冲红。"""
    rng = random.Random(77)
    all_samples = [inv for t in payload.topics for inv in t.samples]
    all_cands = sorted({inv for t in payload.topics for inv in t.candidates})
    batches = []
    for size in (1, 5, 50, 200, 1, 33, 100, 2, 17, 64, 1, 9, 40, 128, 3):
        chosen = rng.sample(all_samples, min(size, len(all_samples)))
        # 混入备选票与清册外票号
        chosen += rng.sample(all_cands, min(size // 2 + 1, len(all_cands)))
        chosen += [f"UNK-{rng.randint(0, 99999):05d}" for _ in range(max(1, size // 10))]
        # 制造批内重复与跨批重复（已退出票再次冲红）
        if chosen:
            chosen.append(chosen[0])
        if batches and rng.random() < 0.5:
            chosen.append(batches[-1][1][0])
        batches.append((f"B{len(batches) + 1:02d}", chosen))
    return batches


def _fresh_conn(tmp_path, name):
    path = str(tmp_path / name)
    os.environ["DB_PATH"] = path
    db.init_db()
    return db.connect()


def _strip_ts(receipt):
    r = dict(receipt)
    r.pop("created_at", None)
    return r


def _dump_state(service_mod, conn, round_id):
    return service_mod.get_round_state(conn, round_id)


def test_new_matches_legacy_receipts_and_state(tmp_path):
    payload = _make_scenario()
    batches = _make_batches(payload)

    old_conn = _fresh_conn(tmp_path, "old.db")
    new_conn = _fresh_conn(tmp_path, "new.db")
    try:
        old_state, old_created = legacy_service.register_round(old_conn, payload)
        new_state, new_created = new_service.register_round(new_conn, payload)
        assert old_created and new_created
        assert old_state == new_state

        old_receipts, new_receipts = [], []
        for batch_no, invoices in batches:
            r_old, replayed_old = legacy_service.process_redflush(
                old_conn, payload.round_id, batch_no, invoices
            )
            r_new, replayed_new = new_service.process_redflush(
                new_conn, payload.round_id, batch_no, invoices
            )
            assert replayed_old == replayed_new
            old_receipts.append(_strip_ts(r_old))
            new_receipts.append(_strip_ts(r_new))

        assert old_receipts == new_receipts
        assert _dump_state(legacy_service, old_conn, payload.round_id) == _dump_state(
            new_service, new_conn, payload.round_id
        )

        # 同批号重送同组票：两边都返回原回执
        b0_no, b0_invs = batches[0]
        old_replay, old_flag = legacy_service.process_redflush(
            old_conn, payload.round_id, b0_no, list(reversed(b0_invs))
        )
        new_replay, new_flag = new_service.process_redflush(
            new_conn, payload.round_id, b0_no, list(reversed(b0_invs))
        )
        assert old_flag and new_flag
        assert _strip_ts(old_replay) == _strip_ts(new_replay) == old_receipts[0]

        # 同批号换组：两边都 409
        with pytest.raises(legacy_service.ApiError) as ei_old:
            legacy_service.process_redflush(old_conn, payload.round_id, b0_no, ["__DIFF__"])
        with pytest.raises(new_service.ApiError) as ei_new:
            new_service.process_redflush(new_conn, payload.round_id, b0_no, ["__DIFF__"])
        assert ei_old.value.status_code == ei_new.value.status_code == 409

        # 底层表内容（active/available/source/position/seq）逐行一致
        for table in ("topics", "samples", "candidates", "batches"):
            old_rows = [tuple(r) for r in old_conn.execute(
                f"SELECT * FROM {table} WHERE round_id=? ORDER BY 1,2,3,4",
                (payload.round_id,),
            ).fetchall()]
            new_rows = [tuple(r) for r in new_conn.execute(
                f"SELECT * FROM {table} WHERE round_id=? ORDER BY 1,2,3,4",
                (payload.round_id,),
            ).fetchall()]
            if table == "batches":
                # 回执 JSON 与行内 created_at 均为当次时间，允许不同，其余结构须一致
                def _norm_batches_row(r):
                    payload_json = json.loads(r[4]) | {"created_at": "X"}
                    return r[:4] + (json.dumps(payload_json, ensure_ascii=False),) + ("X",)

                old_rows = [_norm_batches_row(r) for r in old_rows]
                new_rows = [_norm_batches_row(r) for r in new_rows]
            assert old_rows == new_rows, table
    finally:
        old_conn.close()
        new_conn.close()


def test_new_service_reads_legacy_database(tmp_path):
    """旧库直接可用：旧实现写出来的库，新实现能继续查、重放、接着处理新批次。"""
    payload = _make_scenario(seed=4242)
    batches = _make_batches(payload)[:6]
    conn = _fresh_conn(tmp_path, "legacy.db")
    try:
        legacy_service.register_round(conn, payload)
        for batch_no, invoices in batches:
            legacy_service.process_redflush(conn, payload.round_id, batch_no, invoices)
        legacy_state = legacy_service.get_round_state(conn, payload.round_id)
    finally:
        conn.close()

    # 新实现重新打开同一个数据库文件（模拟升级后重启）
    os.environ["DB_PATH"] = str(tmp_path / "legacy.db")
    conn = db.connect()
    try:
        assert new_service.get_round_state(conn, payload.round_id) == legacy_state
        # 重放旧批次：原回执原样返回
        b0_no, b0_invs = batches[0]
        receipt, replayed = new_service.process_redflush(
            conn, payload.round_id, b0_no, b0_invs
        )
        assert replayed
        stored = new_service.get_receipt(conn, payload.round_id, b0_no)
        assert receipt == stored
        # 接着提交新批次，正常生效且不破坏既有数据
        new_receipt, replayed = new_service.process_redflush(
            conn, payload.round_id, "B-AFTER-UPGRADE",
            [t for t in payload.topics[0].samples[:2]] + ["UNK-NEW"],
        )
        assert not replayed and new_receipt["seq"] == len(batches) + 1
        assert new_service.get_round_state(
            conn, payload.round_id
        )["batches_processed"] == len(batches) + 1
    finally:
        conn.close()
