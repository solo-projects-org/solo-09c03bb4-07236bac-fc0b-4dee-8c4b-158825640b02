#!/usr/bin/env python3
"""月末规模自测：约 8 万条备选关系、单批 1000 张冲红票。

验证点：
1. 服务可直接处理超大批次（票号多到一条 SQL 放不下，走临时表分块）；
2. 回执语义正确：从未登记票判 unknown、已退出票不重复记作本批退出、
   共享备选在各专题分别移除、全批剔除后才补样且每票最多占一个名额；
3. 任一步失败整体回滚、不产生回执；
4. 同轮同批号重送同组票返回原回执、换组报 409；
5. 与优化前旧实现（tests/legacy_baseline）在同场景下的回执/名单一致（除时间）；
6. 对比新旧实现的 SQL 语句数与耗时。

用法: python scripts/load_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.db as db  # noqa: E402
from app import service as new_service  # noqa: E402
from tests.legacy_baseline import legacy_service  # noqa: E402

N_TOPICS = 20
SAMPLES_PER_TOPIC = 200          # 样本 4000
CANDIDATES_PER_TOPIC = 4000      # 备选关系 20 * 4000 = 80000
SHARED_POOL = 12000              # 备选票池，跨专题大量共享
BATCH_SIZE = 1000


class _Topic:
    def __init__(self, topic_code, quota, samples, candidates):
        self.topic_code, self.quota = topic_code, quota
        self.samples, self.candidates = samples, candidates

    def model_dump(self):
        return {"topic_code": self.topic_code, "quota": self.quota,
                "samples": self.samples, "candidates": self.candidates}


class _Payload:
    def __init__(self, round_id, topics):
        self.round_id, self.topics = round_id, topics

    def model_dump(self):
        return {"round_id": self.round_id,
                "topics": [t.model_dump() for t in self.topics]}


def build_roster(round_id: str) -> _Payload:
    pool = [f"K{idx:06d}" for idx in range(SHARED_POOL)]
    topics = []
    for i in range(N_TOPICS):
        samples = [f"S{i:02d}-{j:04d}" for j in range(SAMPLES_PER_TOPIC)]
        start = (i * 137) % (SHARED_POOL - CANDIDATES_PER_TOPIC)
        candidates = pool[start:start + CANDIDATES_PER_TOPIC]
        topics.append(_Topic(f"T{i:02d}", SAMPLES_PER_TOPIC, samples, candidates))
    return _Payload(round_id, topics)


def shared_candidate_invoices(roster: _Payload, need: int) -> list[str]:
    """挑出确实同时出现在 >=2 个专题备选名单上的票号。"""
    owners: dict[str, set] = {}
    for t in roster.topics:
        for inv in t.candidates:
            owners.setdefault(inv, set()).add(t.topic_code)
    multi = sorted(inv for inv, ts in owners.items() if len(ts) >= 2)
    assert len(multi) >= need, f"共享备选不足: {len(multi)} < {need}"
    return multi[:need]


class CountingConn:
    """统计经由 service 发出的 SQL 语句数（executemany 计 1 条）。"""

    def __init__(self, real):
        self._real = real
        self.statements = 0

    def execute(self, sql, params=()):
        self.statements += 1
        return self._real.execute(sql, params)

    def executemany(self, sql, seq):
        self.statements += 1
        return self._real.executemany(sql, seq)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _strip_ts(r):
    r = dict(r)
    r.pop("created_at", None)
    return r


def _check_invariants(state, label: str):
    seen = set()
    for t in state["topics"]:
        assert len(t["samples"]) <= t["quota"], f"{label}: 样本超名额"
        for inv in t["samples"]:
            assert inv not in seen, f"{label}: 票 {inv} 占多个名额"
            seen.add(inv)


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="redflush-load-")
    failures = []

    def check(cond, msg):
        print(("  ✓ " if cond else "  ✗ ") + msg)
        if not cond:
            failures.append(msg)

    def get_state(path):
        os.environ["DB_PATH"] = path
        with db.read_conn() as c:
            return new_service.get_round_state(c, "LOAD")

    # ---------- 新实现：8 万备选关系 ----------
    new_path = os.path.join(workdir, "new.db")
    os.environ["DB_PATH"] = new_path
    db.init_db()
    roster = build_roster("LOAD")
    n_cand_rel = sum(len(t.candidates) for t in roster.topics)
    print(f"[1] 清册规模：{N_TOPICS} 个专题，备选关系 {n_cand_rel} 条，"
          f"样本 {sum(len(t.samples) for t in roster.topics)} 张")
    assert n_cand_rel == 80000

    # 与应用路由一致：写操作在 BEGIN IMMEDIATE 事务内，连接用 CountingConn 包装计数
    sql_new = 0
    t0 = time.perf_counter()
    with db.immediate_tx() as raw:
        conn = CountingConn(raw)
        new_service.register_round(conn, roster)
        sql_new += conn.statements
    t_reg = time.perf_counter() - t0
    print(f"    登记耗时 {t_reg:.3f}s，SQL {sql_new} 条")

    # 单批 1000 张：800 在册样本 + 100 多专题共享备选 + 100 清册外票号
    flush_samples = [f"S{i:02d}-{j:04d}"
                     for i in range(4) for j in range(200)]   # 800
    flush_cands = shared_candidate_invoices(roster, 100)       # 100（确认多专题共享）
    ghosts = [f"GHOST{idx:05d}" for idx in range(100)]         # 100 未知
    batch = flush_samples + flush_cands + ghosts
    assert len(batch) == BATCH_SIZE

    t0 = time.perf_counter()
    with db.immediate_tx() as raw:
        conn = CountingConn(raw)
        receipt, replayed = new_service.process_redflush(conn, "LOAD", "B1000", batch)
        sql_new = conn.statements
    t_batch_new = time.perf_counter() - t0
    print(f"[2] 新实现单批 {BATCH_SIZE} 张：{t_batch_new:.3f}s，SQL {sql_new} 条")

    check(not replayed, "批次首次生效（非重放）")
    check(len(receipt["invoice_nos"]) == BATCH_SIZE, "回执票号数=1000 且已规范化")
    removed_samples = sorted(v for lst in receipt["removed"]["samples"].values() for v in lst)
    check(removed_samples == sorted(flush_samples), "800 张在册样本全部记作本批退出")
    # 100 张共享备选每张出现在多个专题，回执按专题分别列出
    removed_cand_topics = set(receipt["removed"]["candidates"].keys())
    removed_cand_invs = {v for lst in receipt["removed"]["candidates"].values() for v in lst}
    check(removed_cand_invs == set(flush_cands), "100 张共享备选均被移除")
    check(len(removed_cand_topics) > 1,
          f"共享备选在 {len(removed_cand_topics)} 个专题分别列出")
    check(receipt["unknown_invoices"] == sorted(ghosts), "100 张清册外票号判为 unknown")
    check(receipt["total_filled"] > 0, f"全批剔除后已补样 {receipt['total_filled']} 个名额")
    check(all(t["shortfall"] >= 0 for t in receipt["topics"]), "各专题剩余缺额非负")

    state = get_state(new_path)
    _check_invariants(state, "新实现补样后")
    check(state["batches_processed"] == 1, "回执计数=1")
    for tc in removed_cand_topics:
        topic_state = next(t for t in state["topics"] if t["topic_code"] == tc)
        check(not (set(flush_cands) & set(topic_state["candidates"])),
              f"专题 {tc} 的共享备选已从可用名单移除")

    # 重送同组票（换序、含重复）-> 原回执；换组 -> 409
    with db.immediate_tx() as raw:
        conn = CountingConn(raw)
        again, flag = new_service.process_redflush(
            conn, "LOAD", "B1000", list(reversed(batch)) + [batch[0]])
        try:
            new_service.process_redflush(conn, "LOAD", "B1000", ["S00-0001"])
            conflict_ok = False
        except new_service.ApiError as e:
            conflict_ok = e.status_code == 409
    check(flag and _strip_ts(again) == _strip_ts(receipt), "同批号同组票重送返回原回执(200)")
    check(conflict_ok, "同批号换组报 409 冲突")

    # 已退出票再次冲红：不重复记作退出，且仍非 unknown
    with db.immediate_tx() as raw:
        r2, _ = new_service.process_redflush(
            CountingConn(raw), "LOAD", "B2", flush_samples[:50] + flush_cands[:20])
    check(r2["removed"] == {"samples": {}, "candidates": {}},
          "已退出的票不再记作本批退出")
    check(r2["unknown_invoices"] == [], "已登记（含已退出）的票不算 unknown")

    # 失败回滚：与应用路由同一方式包在 immediate 事务里，注入补样阶段异常
    before = get_state(new_path)
    orig = new_service.allocate
    new_service.allocate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("注入失败"))
    raised = False
    try:
        with db.immediate_tx() as raw:
            new_service.process_redflush(
                CountingConn(raw), "LOAD", "BF",
                [f"S05-{j:04d}" for j in range(100)])
    except RuntimeError:
        raised = True
    finally:
        new_service.allocate = orig
    check(raised, "补样阶段异常向上抛出且事务整体回滚")
    after = get_state(new_path)
    check(after == before, "失败后名单原样保留")
    check(after["batches_processed"] == 2, "失败批次不产生回执")
    # 回滚后仍可正常处理新批次（验证临时表/事务状态干净）
    with db.immediate_tx() as raw:
        r3, _ = new_service.process_redflush(
            CountingConn(raw), "LOAD", "BAFTER",
            [f"S06-{j:04d}" for j in range(10)])
    check(r3["seq"] == 3 and get_state(new_path)["batches_processed"] == 3,
          "回滚后新批次正常生效")

    # ---------- 旧实现同场景对照（同样包 immediate 事务，保证公平） ----------
    old_path = os.path.join(workdir, "old.db")
    os.environ["DB_PATH"] = old_path
    db.init_db()
    with db.immediate_tx() as raw:
        legacy_service.register_round(CountingConn(raw), build_roster("LOAD"))
    t0 = time.perf_counter()
    with db.immediate_tx() as raw:
        conn = CountingConn(raw)
        old_receipt, old_flag = legacy_service.process_redflush(
            conn, "LOAD", "B1000", batch)
        sql_old = conn.statements
    t_batch_old = time.perf_counter() - t0

    print(f"[3] 旧实现单批 {BATCH_SIZE} 张：{t_batch_old:.3f}s，SQL {sql_old} 条")
    check(not old_flag, "旧实现同样首次生效")
    check(_strip_ts(old_receipt) == _strip_ts(receipt),
          "除 created_at 外回执与优化前完全一致")
    old_state = get_state(old_path)
    check(old_state == state, "现行名单与优化前完全一致")
    check(sql_new < sql_old, f"SQL 条数 {sql_new} < 旧实现 {sql_old}（不再逐票执行）")

    # ---------- 更大批次（3000 张，多分块） ----------
    with db.immediate_tx() as raw:
        big = [f"S07-{j:04d}" for j in range(200)] + [f"X{idx:05d}" for idx in range(2800)]
        rb, flag = new_service.process_redflush(CountingConn(raw), "LOAD", "B3000", big)
    check(not flag and len(rb["unknown_invoices"]) == 2800,
          "3000 张批次（跨 6 个分块）正常处理，未知票判定正确")
    _check_invariants(get_state(new_path), "3000 批后")

    print("\n================ 自测结果 ================")
    print(f"备选关系规模：{n_cand_rel} 条；单批冲红：{BATCH_SIZE} 张（另测 3000 张）")
    print(f"新实现批处理：{t_batch_new:.3f}s / {sql_new} 条 SQL；"
          f"旧实现：{t_batch_old:.3f}s / {sql_old} 条 SQL")
    print(f"SQL 降幅：{(1 - sql_new / sql_old) * 100:.1f}%；"
          f"耗时：{t_batch_new:.3f}s vs {t_batch_old:.3f}s")
    if failures:
        print(f"未通过 {len(failures)} 项：")
        for m in failures:
            print("  - " + m)
        return 1
    print("全部断言通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
