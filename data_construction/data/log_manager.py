from typing import TypedDict,List,Any,Dict,Iterator,Optional,Set,Literal
from dataclasses import dataclass,asdict,field
from datasets import load_dataset
from pathlib import Path
import os
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

@dataclass
class InteractionItem:
    agent: str
    response_metadata: Dict[str, int]
    content: Dict

@dataclass
class LogItem:
    # === ID ===
    log_id: Optional[int] = field(default=None)
    # === 任务信息 ===
    status: Literal["success", "fail"] = field(default="success")
    task: Optional[str] = field(default=None)
    fail_reason: Optional[str] = field(default=None)
    # === 数据信息 === 
    mt_id: Optional[str] = field(default=-1)  # -1表示失败
    hash_id: Optional[str] = field(default=None)
    source: Optional[str] = field(default=None)
    # === 交互信息 ===
    interactions: Optional[List[InteractionItem]] = field(default=None)
    # === 聚合信息 ===
    total_prompt_tokens: Optional[int] = field(default=0)
    total_completion_tokens: Optional[int] = field(default=0)
    total_tokens: Optional[int] = field(default=0)
    interaction_number: Optional[int] = field(default=0)
    # === 时间信息 ===
    created_at: Optional[datetime] = field(default=None)
    updated_at: Optional[datetime] = field(default=None)

class LogManager:
    """
    日志管理类
    """
    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            db_path = (Path(__file__).parent / "logs.db").resolve()
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # 如果不存在才进行初始化
        if not db_path.exists():
            self._init_db()

    @contextmanager
    def get_cursor(self):
        """上下文管理器：提供数据库连接和自动提交/回滚"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 支持通过列名访问
        try:
            yield conn.cursor()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """初始化数据库表结构"""
        with self.get_cursor() as cur:
            # 主表：日志记录
            cur.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL CHECK(status IN ('success', 'fail')),
                    task TEXT NOT NULL,
                    fail_reason TEXT,
                    mt_id INTEGER NOT NULL,
                    hash_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    interactions_json TEXT NOT NULL,
                    total_prompt_tokens INTEGER NOT NULL,
                    total_completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    interaction_number INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 索引优化查询
            cur.execute("CREATE INDEX IF NOT EXISTS idx_logs_hash_id ON logs (hash_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_logs_mt_id ON logs (mt_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_logs_status ON logs (status)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_logs_task ON logs (task)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_logs_created_at ON logs (created_at)")

    # =========================
    # 增删改查接口
    # =========================

    def add(self, log_item: LogItem) -> int:
        """
        添加一个新的日志记录
        返回插入的 log_id
        """
        interactions_json = json.dumps([asdict(interaction) for interaction in (log_item.interactions or [])], ensure_ascii=False)
        with self.get_cursor() as cur:
            cur.execute("""
                INSERT INTO logs 
                (status, task, fail_reason, mt_id, hash_id, source, interactions_json, 
                 total_prompt_tokens, total_completion_tokens, total_tokens, interaction_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log_item.status,
                log_item.task,
                log_item.fail_reason,
                log_item.mt_id,
                log_item.hash_id,
                log_item.source,
                interactions_json,
                log_item.total_prompt_tokens,
                log_item.total_completion_tokens,
                log_item.total_tokens,
                log_item.interaction_number
            ))
            return cur.lastrowid

    def get(self, log_id: int) -> Optional[LogItem]:
        """根据 log_id 查询一条日志记录"""
        with self.get_cursor() as cur:
            cur.execute("SELECT * FROM logs WHERE log_id = ?", (log_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_log_item(row)

    def update(self, log_item: LogItem) -> bool:
        """更新一个已存在的日志记录（必须存在）"""
        if log_item.log_id is None or not self.exists(log_item.log_id):
            return False
        interactions_json = json.dumps([asdict(interaction) for interaction in (log_item.interactions or [])], ensure_ascii=False)
        with self.get_cursor() as cur:
            cur.execute("""
                UPDATE logs SET
                status = ?, task = ?, fail_reason = ?, mt_id = ?, hash_id = ?, source = ?,
                interactions_json = ?, total_prompt_tokens = ?,
                total_completion_tokens = ?, total_tokens = ?, interaction_number = ?,
                updated_at = CURRENT_TIMESTAMP
                WHERE log_id = ?
            """, (
                log_item.status,
                log_item.task,
                log_item.fail_reason,
                log_item.mt_id,
                log_item.hash_id,
                log_item.source,
                interactions_json,
                log_item.total_prompt_tokens,
                log_item.total_completion_tokens,
                log_item.total_tokens,
                log_item.interaction_number,
                log_item.log_id
            ))
            return cur.rowcount > 0

    def delete(self, log_id: int) -> bool:
        """删除指定 log_id 的日志记录"""
        with self.get_cursor() as cur:
            cur.execute("DELETE FROM logs WHERE log_id = ?", (log_id,))
            return cur.rowcount > 0

    def exists(self, log_id: int) -> bool:
        """检查 log_id 是否已存在"""
        with self.get_cursor() as cur:
            cur.execute("SELECT 1 FROM logs WHERE log_id = ? LIMIT 1", (log_id,))
            return cur.fetchone() is not None

    def get_all_log_ids(self) -> Set[int]:
        """返回所有已存在的 log_id 集合"""
        with self.get_cursor() as cur:
            cur.execute("SELECT log_id FROM logs")
            return {row["log_id"] for row in cur.fetchall()}

    def list_all(self) -> List[LogItem]:
        """获取所有日志记录"""
        with self.get_cursor() as cur:
            cur.execute("SELECT * FROM logs ORDER BY created_at DESC")
            return [self._row_to_log_item(row) for row in cur.fetchall()]

    def list_by_hash_id(self, hash_id: str) -> List[LogItem]:
        """根据 hash_id 查询日志记录"""
        with self.get_cursor() as cur:
            cur.execute("SELECT * FROM logs WHERE hash_id = ? ORDER BY created_at DESC", (hash_id,))
            return [self._row_to_log_item(row) for row in cur.fetchall()]

    def list_by_mt_id(self, mt_id: int) -> List[LogItem]:
        """根据 mt_id 查询日志记录"""
        with self.get_cursor() as cur:
            cur.execute("SELECT * FROM logs WHERE mt_id = ? ORDER BY created_at DESC", (mt_id,))
            return [self._row_to_log_item(row) for row in cur.fetchall()]

    def list_by_status(self, status: Literal["success", "fail"]) -> List[LogItem]:
        """根据状态查询日志记录"""
        with self.get_cursor() as cur:
            cur.execute("SELECT * FROM logs WHERE status = ? ORDER BY created_at DESC", (status,))
            return [self._row_to_log_item(row) for row in cur.fetchall()]

    def list_by_task(self, task: str) -> List[LogItem]:
        """根据任务名称查询日志记录"""
        with self.get_cursor() as cur:
            cur.execute("SELECT * FROM logs WHERE task = ? ORDER BY created_at DESC", (task,))
            return [self._row_to_log_item(row) for row in cur.fetchall()]

    # =========================
    # 导入导出功能
    # =========================

    def setup_with_jsonl(self, jsonl_path: Path) -> None:
        """
        从 JSONL 文件初始化数据库：清空旧数据，导入新数据
        每行是一个 LogItem 的 dict 结构
        """
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        # 清空现有数据
        with self.get_cursor() as cur:
            cur.execute("DELETE FROM logs")

        # 逐行导入
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    log_item = self._dict_to_log_item(data)
                    self.add(log_item)
                except Exception as e:
                    raise ValueError(f"Invalid JSONL format at line {line_num}: {e}")

    def export_jsonl(self, output_path: Path) -> None:
        """
        将数据库中所有数据导出为 JSONL 文件
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        log_items = self.list_all()
        with open(output_path, 'w', encoding='utf-8') as f:
            for log_item in log_items:
                # 修复datetime无法序列化的问题
                f.write(json.dumps(self._log_item_to_dict(log_item), ensure_ascii=False) + '\n')

    # =========================
    # 内部辅助方法
    # =========================

    def _log_item_to_dict(self, log_item: LogItem) -> Dict[str, Any]:
        """将 LogItem 转换为可 JSON 序列化的字典"""
        result = asdict(log_item)
        # 处理 datetime 对象
        for field in ['created_at', 'updated_at']:
            if result[field] is not None:
                result[field] = result[field].isoformat()
        return result

    def _row_to_log_item(self, row: sqlite3.Row) -> LogItem:
        """将数据库行转换为 LogItem 对象"""
        interactions = [
            InteractionItem(**interaction_data)
            for interaction_data in json.loads(row["interactions_json"])
        ]
        return LogItem(
            log_id=row["log_id"],
            status=row["status"],
            task=row["task"],
            fail_reason=row["fail_reason"],
            mt_id=row["mt_id"],
            hash_id=row["hash_id"],
            source=row["source"],
            interactions=interactions,
            total_prompt_tokens=row["total_prompt_tokens"],
            total_completion_tokens=row["total_completion_tokens"],
            total_tokens=row["total_tokens"],
            interaction_number=row["interaction_number"],
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
            updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
        )

    def _dict_to_log_item(self, data: Dict[str, Any]) -> LogItem:
        """将 dict 转换为 LogItem"""
        interactions = [InteractionItem(**interaction_data) for interaction_data in data.get("interactions", [])]
        return LogItem(
            log_id=data.get("log_id"),
            status=data["status"],
            task=data.get("task"),
            fail_reason=data.get("fail_reason"),
            mt_id=data.get("mt_id"),
            hash_id=data.get("hash_id"),
            source=data.get("source"),
            interactions=interactions,
            total_prompt_tokens=data.get("total_prompt_tokens", 0),
            total_completion_tokens=data.get("total_completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            interaction_number=data.get("interaction_number", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
        )

    # =========================
    # 工具方法
    # =========================

    def count(self) -> int:
        """返回当前日志总量"""
        with self.get_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM logs")
            return cur.fetchone()[0]

    def count_by_status(self, status: Literal["success", "fail"]) -> int:
        """返回指定状态的日志数量"""
        with self.get_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM logs WHERE status = ?", (status,))
            return cur.fetchone()[0]

    def count_by_hash_id(self, hash_id: str) -> int:
        """返回指定 hash_id 的日志数量"""
        with self.get_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM logs WHERE hash_id = ?", (hash_id,))
            return cur.fetchone()[0]

    def clear(self) -> None:
        """清空所有日志数据"""
        with self.get_cursor() as cur:
            cur.execute("DELETE FROM logs")

    def close(self):
        """关闭资源（当前无长期连接）"""
        pass