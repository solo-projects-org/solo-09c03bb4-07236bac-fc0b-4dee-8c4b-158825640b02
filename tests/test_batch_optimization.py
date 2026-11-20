"""批量优化专项：分块临时表、超大批次、回执语义、失败回滚、SQL 次数不随票数膨胀。"""
import pytest

import app.db as db
from app import service
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    db.init_db()
    with TestClient(app) as c:
        yield c


def _big_roster(round_id="RB", n_topics=10, per_topic=300):
    """构造约 3000 条备选关系的清册，备选跨专题大量共享。"""
    shared = [f"K{idx:05d}" for idx in range(1200)]
    topics = []
    for i in range(n_topics):
        samples = [f"S{i}-{j:04d}" for j in range(20)]
        candidates = shared[i * per_topic % len(shared):][:per_topic]
        topics.append(
            {"topic_code": f"T{i}", "quota": 20, "samples": samples, "candidates": candidates}
        )
    return {"round_id": round_id, "topics": topics}


def test_batch_larger_than_sql_variable_limit(client):
    """单批 1000 张冲红票（超过 SQLite 999 变量上限）仍能一次提交并正确处理。"""
    roster = _big_roster()
    r = client.post("/rounds", json=roster)
    assert r.status_code == 201, r.text

    known_samples = [f"S{i}-{j:04d}" for i in range(10) for j in range(20)]  # 200 个在册样本
    invoices = known_samples + [f"GHOST{idx:04d}" for idx in range(800)]     # 共 1000 张
    assert len(invoices) == 1000
    r = client.post("/rounds/RB/redflush-batches",
                    json={"batch_no": "BIG1", "invoice_nos": invoices})
    assert r.status_code == 201, r.text
    receipt = r.json()
    assert len(receipt["invoice_nos"]) == 1000
    removed = sorted(inv for lst in receipt["removed"]["samples"].values() for inv in lst)
    assert removed == sorted(known_samples)          # 200 个在册样本全部退出
    assert receipt["unknown_invoices"] == sorted(f"GHOST{idx:04d}" for idx in range(800))
    # 全批剔除后已补样，且每张票最多占一个名额
    state = client.get("/rounds/RB").json()
    seen = set()
    for t in state["topics"]:
        assert len(t["samples"]) <= t["quota"]
        for inv in t["samples"]:
            assert inv not in seen
            seen.add(inv)


def test_chunking_works_above_and_below_chunk_size(client, monkeypatch):
    """500/501/1500 张等不同规模（覆盖 1、2、3 个分块）行为与票号集合一致。"""
    monkeypatch.setattr(service, "CHUNK_SIZE", 500)
    roster = _big_roster("RC")
    assert client.post("/rounds", json=roster).status_code == 201
    pool = [f"S{i}-{j:04d}" for i in range(10) for j in range(20)]
    for n in (1, 499, 500, 501, 1500):
        invs = (pool * ((n // len(pool)) + 1))[:n]
        r = client.post("/rounds/RC/redflush-batches",
                        json={"batch_no": f"B{n}", "invoice_nos": invs})
        assert r.status_code == 201, (n, r.text)
        assert len(r.json()["invoice_nos"]) == len(set(invs))


def test_already_exited_invoices_not_recounted(client):
    """已退出的票再次出现在本批：不记作本批退出；从未登记的票仍按清册判未知。"""
    client.post("/rounds", json={
        "round_id": "RX",
        "topics": [
            {"topic_code": "T1", "quota": 2, "samples": ["A1", "A2"], "candidates": ["C1", "C2"]},
            {"topic_code": "T2", "quota": 1, "samples": ["B1"], "candidates": ["C1", "C3"]},
        ],
    })
    first = client.post("/rounds/RX/redflush-batches",
                        json={"batch_no": "B1", "invoice_nos": ["A1", "C1"]}).json()
    assert first["removed"]["samples"] == {"T1": ["A1"]}
    # C1 同时在 T1、T2 的可用备选名单上：各专题分别列出并移除
    assert first["removed"]["candidates"] == {"T1": ["C1"], "T2": ["C1"]}

    second = client.post("/rounds/RX/redflush-batches",
                         json={"batch_no": "B2",
                               "invoice_nos": ["A1", "C1", "NOPE"]}).json()
    # A1、C1 此前已退出，不再记作本批退出；NOPE 从未登记 -> unknown
    assert second["removed"] == {"samples": {}, "candidates": {}}
    assert second["unknown_invoices"] == ["NOPE"]
    # 名单不变（除 B1 已造成的变化）
    state = client.get("/rounds/RX").json()
    t1 = next(t for t in state["topics"] if t["topic_code"] == "T1")
    t2 = next(t for t in state["topics"] if t["topic_code"] == "T2")
    assert "C1" not in t1["candidates"] and "C1" not in t2["candidates"]
    assert "A1" not in t1["samples"]


def test_rollback_on_failure_keeps_roster_and_no_receipt(client, monkeypatch):
    """全批剔除后、补样阶段失败：整批回滚，原名单保留且不产生回执。"""
    client.post("/rounds", json=_big_roster("RR"))
    before = client.get("/rounds/RR").json()

    orig_allocate = service.allocate

    def boom(*_a, **_k):
        raise RuntimeError("模拟匹配阶段失败")

    monkeypatch.setattr(service, "allocate", boom)
    invoices = [f"S{i}-{j:04d}" for i in range(10) for j in range(15)][:1000]
    with pytest.raises(RuntimeError):
        client.post("/rounds/RR/redflush-batches",
                    json={"batch_no": "WILLFAIL", "invoice_nos": invoices})
    after = client.get("/rounds/RR").json()
    assert after == before  # 名单原样
    assert after["batches_processed"] == 0  # 无回执
    # 手动恢复（不能用 monkeypatch.undo()：会连带撤销 fixture 设置的 DB_PATH）
    service.allocate = orig_allocate
    # 临时表随回滚自动清理，同库后续批次可正常处理
    ok = client.post("/rounds/RR/redflush-batches",
                     json={"batch_no": "OK1", "invoice_nos": invoices[:10]})
    assert ok.status_code == 201, ok.text


def test_sql_statement_count_independent_of_batch_size(tmp_path, monkeypatch):
    """SQL 语句数为分块常数级，不随批内票数线性增长（对比旧实现逐票 SQL）。"""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "cnt.db"))
    db.init_db()

    class CountingConn:
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

    conn = CountingConn(db.connect())
    payload = _PayloadLike(_big_roster("RN")["topics"], "RN")
    service.register_round(conn, payload)
    conn.statements = 0
    service.process_redflush(
        conn, "RN", "B100", [f"S{i}-{j:04d}" for i in range(10) for j in range(10)]
    )
    sql_100 = conn.statements
    conn.statements = 0
    service.process_redflush(
        conn, "RN", "B1000", [f"S{i}-{j:04d}" for i in range(10) for j in range(20)][:200]
    )
    sql_200 = conn.statements
    # 票数翻倍，SQL 条数基本持平（仅临时表分块插入按 500/块，两批都只 1 块）
    assert sql_200 <= sql_100 + 2, (sql_100, sql_200)
    # 且绝对数量远低于"逐票 SQL"：旧实现 100 票就需要 400+ 条
    assert sql_100 < 40, sql_100
    conn.close()


class _TopicLike:
    def __init__(self, d):
        self.__dict__.update(d)

    def model_dump(self):
        return dict(self.__dict__)


class _PayloadLike:
    def __init__(self, topic_dicts, round_id):
        self.round_id = round_id
        self.topics = [_TopicLike(d) for d in topic_dicts]

    def model_dump(self):
        return {"round_id": self.round_id,
                "topics": [t.model_dump() for t in self.topics]}
