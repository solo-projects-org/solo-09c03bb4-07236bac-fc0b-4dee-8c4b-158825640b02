"""业务逻辑：登记清册、处理冲红批、查询现行名单与回执。

所有写函数都必须在 db.immediate_tx() 事务内调用：
查重 -> 整批退出 -> 计算缺额 -> 最大匹配补样 -> 写回执，要么全部生效，要么整体回滚。

月末可能单批上千张冲红票，本模块不再逐票执行 SQL：
- 本批票号经排序去重后分块（CHUNK_SIZE）写入事务级临时表，
  后续退出/判定/补样全部以集合（JOIN/IN 临时表）方式批量查写，
  一条 SQL 装不下的批次由分块写入兜底，SQL 次数与批内票数无关；
- 清册查询按专题一次性聚合，不再对每张票反复扫描整轮名单。
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from .matching import allocate

# 单条 SQL 的参数上限取保守值：SQLite 默认可用 999 个变量，
# 临时表分块写入每块 500 行，保证任何批次（含单批 1000+ 张）都能放下。
CHUNK_SIZE = 500


class ApiError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canon_key(invoices) -> str:
    """票号集合规范化：去重排序，保证同组票号（不论顺序）判定为同一批。"""
    return "\n".join(sorted(set(invoices)))


def _roster_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _chunks(seq, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _load_batch_temp_table(conn, canon: list[str]) -> str:
    """把本批规范化票号分块写入事务级临时表，返回表名。

    临时表仅当前连接可见；写在同一事务里，事务回滚时其中数据随之消失。
    分块 INSERT 使 SQL 语句数为 ceil(n/CHUNK_SIZE)+常数级，与逐票无关，
    且不受单条 SQL 参数数量限制。
    """
    table = "tmp_redflush_invoices"
    conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(f"CREATE TEMP TABLE {table} (invoice_no TEXT PRIMARY KEY)")
    for part in _chunks(canon, CHUNK_SIZE):
        conn.executemany(
            f"INSERT INTO {table}(invoice_no) VALUES (?)", [(inv,) for inv in part]
        )
    return table


# ---------------------------------------------------------------- 登记清册

def register_round(conn, payload) -> tuple[dict, bool]:
    """登记一轮清册，返回 (现行名单, 是否新建)。

    同轮次同内容重复登记 -> 返回现行名单（幂等）；内容不同 -> 409。
    """
    round_id = payload.round_id
    digest = _roster_hash(payload.model_dump())

    row = conn.execute("SELECT roster_hash FROM rounds WHERE round_id = ?", (round_id,)).fetchone()
    if row:
        if row["roster_hash"] == digest:
            return get_round_state(conn, round_id), False
        raise ApiError(409, f"轮次 {round_id} 已登记，且清册内容不同")

    # ---- 校验：每张票只能占一个名额 ----
    seen_topics = set()
    sample_owner = {}
    for t in payload.topics:
        if t.topic_code in seen_topics:
            raise ApiError(400, f"专题代码重复: {t.topic_code}")
        seen_topics.add(t.topic_code)
        if len(set(t.samples)) != len(t.samples):
            raise ApiError(400, f"专题 {t.topic_code} 的样本票号重复")
        if len(set(t.candidates)) != len(t.candidates):
            raise ApiError(400, f"专题 {t.topic_code} 的备选票号重复")
        if len(t.samples) > t.quota:
            raise ApiError(400, f"专题 {t.topic_code} 的已发样本数超过名额")
        for inv in t.samples:
            if inv in sample_owner:
                raise ApiError(400, f"票号 {inv} 同时是多个专题的样本")
            sample_owner[inv] = t.topic_code
    for t in payload.topics:
        for inv in t.candidates:
            if inv in sample_owner:
                raise ApiError(400, f"票号 {inv} 既是已发样本又是备选")

    # ---- 写入（批量 executemany，避免逐行 SQL）----
    conn.execute(
        "INSERT INTO rounds(round_id, roster_hash, created_at) VALUES (?,?,?)",
        (round_id, digest, _now()),
    )
    conn.executemany(
        "INSERT INTO topics(round_id, topic_code, quota, position) VALUES (?,?,?,?)",
        [(round_id, t.topic_code, t.quota, i) for i, t in enumerate(payload.topics)],
    )
    for i, t in enumerate(payload.topics):
        conn.executemany(
            "INSERT INTO samples(round_id, topic_code, invoice_no, active, source, position)"
            " VALUES (?,?,?,1,'roster',?)",
            [(round_id, t.topic_code, inv, j) for j, inv in enumerate(t.samples)],
        )
        conn.executemany(
            "INSERT INTO candidates(round_id, topic_code, invoice_no, available, position)"
            " VALUES (?,?,?,1,?)",
            [(round_id, t.topic_code, inv, j) for j, inv in enumerate(t.candidates)],
        )
    return get_round_state(conn, round_id), True


# ---------------------------------------------------------------- 冲红批处理

def process_redflush(conn, round_id: str, batch_no: str, invoice_nos) -> tuple[dict, bool]:
    """处理一个冲红批，返回 (回执, 是否重放)。

    同轮同批号：票号集合相同 -> 返回原回执；不同 -> 409。
    整批处理包在一个 immediate 事务内：任一步失败整体回滚，名单不变、不产生回执。
    """
    if not conn.execute("SELECT 1 FROM rounds WHERE round_id=?", (round_id,)).fetchone():
        raise ApiError(404, f"轮次不存在: {round_id}")

    canon = sorted(set(invoice_nos))
    key = _canon_key(canon)

    row = conn.execute(
        "SELECT invoice_key, receipt_json FROM batches WHERE round_id=? AND batch_no=?",
        (round_id, batch_no),
    ).fetchone()
    if row:
        if row["invoice_key"] == key:
            return json.loads(row["receipt_json"]), True
        raise ApiError(409, f"批号 {batch_no} 在本轮已生效，且票号集合不同")

    topics = conn.execute(
        "SELECT topic_code, quota FROM topics WHERE round_id=? ORDER BY position", (round_id,)
    ).fetchall()
    order = [t["topic_code"] for t in topics]
    quota_by_topic = {t["topic_code"]: t["quota"] for t in topics}

    # 本批票号进事务级临时表，后续全部按集合批量查写（分块写入，不受 SQL 参数上限约束）
    tmp = _load_batch_temp_table(conn, canon)
    # 注意：异常路径不主动 DROP——此时外层事务即将 ROLLBACK，失败事务里再发 DDL
    # 会二次抛错；临时表随 ROLLBACK/连接关闭自动清理，不会泄漏到后续批次。
    receipt = _process_redflush_batch(
        conn, round_id, batch_no, canon, key, tmp, order, quota_by_topic
    )
    conn.execute(f"DROP TABLE IF EXISTS {tmp}")
    return receipt, False


def _process_redflush_batch(conn, round_id, batch_no, canon, key, tmp, order, quota_by_topic):
    # 1) 整批退出样本名单与备选名单。
    #    只统计"当前仍有效"的条目：此前批次已退出的票不再记作本批退出。
    #    同一张冲红票若在多个专题的可用备选名单上，会在各专题分别列出并一并移除。
    removed_samples: dict[str, list[str]] = {}
    removed_cands: dict[str, list[str]] = {}

    sample_rows = conn.execute(
        "SELECT s.topic_code AS topic_code, s.invoice_no AS invoice_no"
        " FROM samples AS s JOIN " + tmp + " AS b ON s.invoice_no = b.invoice_no"
        " WHERE s.round_id=? AND s.active=1"
        " ORDER BY s.topic_code, s.invoice_no",
        (round_id,),
    ).fetchall()
    for r in sample_rows:
        removed_samples.setdefault(r["topic_code"], []).append(r["invoice_no"])

    cand_rows = conn.execute(
        "SELECT c.topic_code AS topic_code, c.invoice_no AS invoice_no"
        " FROM candidates AS c JOIN " + tmp + " AS b ON c.invoice_no = b.invoice_no"
        " WHERE c.round_id=? AND c.available=1"
        " ORDER BY c.topic_code, c.invoice_no",
        (round_id,),
    ).fetchall()
    for r in cand_rows:
        removed_cands.setdefault(r["topic_code"], []).append(r["invoice_no"])

    # 回执按清册判断哪些票从未登记（样本、备选里都不存在，含早已退出的历史票）
    known_rows = conn.execute(
        "SELECT DISTINCT s.invoice_no AS invoice_no FROM samples AS s"
        " JOIN " + tmp + " AS b ON s.invoice_no = b.invoice_no WHERE s.round_id=?"
        " UNION"
        " SELECT DISTINCT c.invoice_no AS invoice_no FROM candidates AS c"
        " JOIN " + tmp + " AS b ON c.invoice_no = b.invoice_no WHERE c.round_id=?",
        (round_id, round_id),
    ).fetchall()
    known = {r["invoice_no"] for r in known_rows}

    # 批量退出：一条 UPDATE 借临时表覆盖整批票号（含历史已退出票，UPDATE 对其无实际效果）
    conn.execute(
        "UPDATE samples SET active=0"
        " WHERE round_id=? AND invoice_no IN (SELECT invoice_no FROM " + tmp + ")",
        (round_id,),
    )
    conn.execute(
        "UPDATE candidates SET available=0"
        " WHERE round_id=? AND invoice_no IN (SELECT invoice_no FROM " + tmp + ")",
        (round_id,),
    )

    # 2) 各专题当前缺额与可用备选：每类名单整轮各扫一次，按专题聚合
    active_counts = dict(
        conn.execute(
            "SELECT topic_code, COUNT(*) AS c FROM samples"
            " WHERE round_id=? AND active=1 GROUP BY topic_code",
            (round_id,),
        ).fetchall()
    )
    vacancies = {
        tc: max(0, quota_by_topic[tc] - active_counts.get(tc, 0)) for tc in order
    }
    cand: dict[str, list[str]] = {tc: [] for tc in order}
    for r in conn.execute(
        "SELECT topic_code, invoice_no FROM candidates"
        " WHERE round_id=? AND available=1 ORDER BY topic_code, position",
        (round_id,),
    ).fetchall():
        cand[r["topic_code"]].append(r["invoice_no"])

    # 3) 全批剔除完成后再补样：最大匹配保证本批补上的名额总数最大
    assign = allocate(order, vacancies, cand)

    # 4) 落库：补入样本名单、消耗掉的备选在所有专题中同时失效
    seq = conn.execute(
        "SELECT COALESCE(MAX(seq),0)+1 AS s FROM batches WHERE round_id=?", (round_id,)
    ).fetchone()["s"]

    next_pos = dict(
        conn.execute(
            "SELECT topic_code, COALESCE(MAX(position),-1)+1 AS p FROM samples"
            " WHERE round_id=? GROUP BY topic_code",
            (round_id,),
        ).fetchall()
    )
    sample_inserts = []
    consumed = []
    for tc in order:
        pos = next_pos.get(tc, 0)
        for k, inv in enumerate(assign[tc]):
            sample_inserts.append((round_id, tc, inv, f"batch:{batch_no}", pos + k))
            consumed.append(inv)
    conn.executemany(
        "INSERT INTO samples(round_id, topic_code, invoice_no, active, source, position)"
        " VALUES (?,?,?,1,?,?)",
        sample_inserts,
    )
    if consumed:
        # 被占用的备选票同样经临时表分块承载，避免超长 IN 列表。
        # 与主临时表同理：异常路径不主动 DROP，交由 ROLLBACK/连接关闭自动清理。
        ctmp = "tmp_consumed_invoices"
        conn.execute(f"DROP TABLE IF EXISTS {ctmp}")
        conn.execute(f"CREATE TEMP TABLE {ctmp} (invoice_no TEXT PRIMARY KEY)")
        for part in _chunks(sorted(set(consumed)), CHUNK_SIZE):
            conn.executemany(
                f"INSERT INTO {ctmp}(invoice_no) VALUES (?)", [(inv,) for inv in part]
            )
        conn.execute(
            "UPDATE candidates SET available=0"
            " WHERE round_id=? AND invoice_no IN (SELECT invoice_no FROM " + ctmp + ")",
            (round_id,),
        )
        conn.execute(f"DROP TABLE IF EXISTS {ctmp}")

    # 5) 回执：结构与优化前完全一致（仅 created_at 为当次时间）
    receipt = {
        "round_id": round_id,
        "batch_no": batch_no,
        "seq": seq,
        "invoice_nos": canon,
        "unknown_invoices": [i for i in canon if i not in known],
        "removed": {
            "samples": {tc: removed_samples[tc] for tc in order if tc in removed_samples},
            "candidates": {tc: removed_cands[tc] for tc in order if tc in removed_cands},
        },
        "topics": [
            {
                "topic_code": tc,
                "vacancy_before": vacancies[tc],
                "replacements": assign[tc],
                "shortfall": vacancies[tc] - len(assign[tc]),
            }
            for tc in order
        ],
        "total_filled": sum(len(v) for v in assign.values()),
        "created_at": _now(),
    }
    conn.execute(
        "INSERT INTO batches(round_id, batch_no, seq, invoice_key, receipt_json, created_at)"
        " VALUES (?,?,?,?,?,?)",
        (round_id, batch_no, seq, key, json.dumps(receipt, ensure_ascii=False), receipt["created_at"]),
    )
    return receipt


# ---------------------------------------------------------------- 查询

def get_round_state(conn, round_id: str) -> dict:
    if not conn.execute("SELECT 1 FROM rounds WHERE round_id=?", (round_id,)).fetchone():
        raise ApiError(404, f"轮次不存在: {round_id}")
    topics = conn.execute(
        "SELECT topic_code, quota FROM topics WHERE round_id=? ORDER BY position", (round_id,)
    ).fetchall()
    samples_by_topic: dict[str, list[str]] = {t["topic_code"]: [] for t in topics}
    cands_by_topic: dict[str, list[str]] = {t["topic_code"]: [] for t in topics}
    # 两张名单各整轮扫描一次，按登记顺序聚合，避免逐专题重复扫描
    for r in conn.execute(
        "SELECT topic_code, invoice_no FROM samples"
        " WHERE round_id=? AND active=1 ORDER BY topic_code, position",
        (round_id,),
    ).fetchall():
        samples_by_topic[r["topic_code"]].append(r["invoice_no"])
    for r in conn.execute(
        "SELECT topic_code, invoice_no FROM candidates"
        " WHERE round_id=? AND available=1 ORDER BY topic_code, position",
        (round_id,),
    ).fetchall():
        cands_by_topic[r["topic_code"]].append(r["invoice_no"])
    out = []
    for t in topics:
        tc = t["topic_code"]
        samples = samples_by_topic[tc]
        out.append(
            {
                "topic_code": tc,
                "quota": t["quota"],
                "samples": samples,
                "candidates": cands_by_topic[tc],
                "shortfall": max(0, t["quota"] - len(samples)),
            }
        )
    nb = conn.execute(
        "SELECT COUNT(*) AS c FROM batches WHERE round_id=?", (round_id,)
    ).fetchone()["c"]
    return {"round_id": round_id, "topics": out, "batches_processed": nb}


def get_receipt(conn, round_id: str, batch_no: str) -> dict:
    row = conn.execute(
        "SELECT receipt_json FROM batches WHERE round_id=? AND batch_no=?",
        (round_id, batch_no),
    ).fetchone()
    if not row:
        raise ApiError(404, f"回执不存在: {round_id}/{batch_no}")
    return json.loads(row["receipt_json"])


def list_receipts(conn, round_id: str) -> list[dict]:
    if not conn.execute("SELECT 1 FROM rounds WHERE round_id=?", (round_id,)).fetchone():
        raise ApiError(404, f"轮次不存在: {round_id}")
    return [
        json.loads(r["receipt_json"])
        for r in conn.execute(
            "SELECT receipt_json FROM batches WHERE round_id=? ORDER BY seq", (round_id,)
        ).fetchall()
    ]
