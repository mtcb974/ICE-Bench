from typing import TypedDict,List,Any,Dict,Iterator,Optional,Set
from dataclasses import dataclass,asdict,field
from datasets import load_dataset
from pathlib import Path
import os
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

@dataclass
class PerTurnDataInstance:
    turn_num: int
    instruction: str
    solution: Optional[str] = field(default=None)
    test: Optional[str] = field(default=None)

@dataclass
class MultiTurnDataInstance:
    hash_id: str    # use for checkpoint
    total_turn: int
    turn_datas: List[PerTurnDataInstance]
    metadata: Dict[str,Any]
    mt_id: Optional[int] = field(default=None)     # 作为数据库主键进行自增

class MTDataManager:
    """
    多轮数据管理类
    """
    def __init__(self,db_path: Optional[Path]= None) -> None:
        if db_path is None:
            db_path = (Path(__file__).parent / "mtcodebench.db").resolve()
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # 只有当数据库文件不存在时才初始化表结构
        if not self.db_path.exists():
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
            # 主表：多轮数据实例
            cur.execute("""
                CREATE TABLE IF NOT EXISTS multi_turn_instances (
                    hash_id TEXT UNIQUE NOT NULL,
                    mt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_turn INTEGER NOT NULL,
                    turn_datas_json TEXT NOT NULL,  -- JSON 字符串存储 List[PerTurnDataInstance]
                    metadata_json TEXT NOT NULL,    -- JSON 字符串存储 Dict
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 索引优化查询
            cur.execute("CREATE INDEX IF NOT EXISTS idx_hash_id ON multi_turn_instances (hash_id)")
        
    # =========================
    # 增删改查接口
    # =========================

    def add(self,instance:MultiTurnDataInstance) -> None:
        """
        添加或更新一个 MultiTurnDataInstance
        以 hash_id 作为唯一键，mt_id 为自增主键，无需显式插入
        """
        turn_datas_json = json.dumps([asdict(t) for t in instance.turn_datas], ensure_ascii=False)
        metadata_json = json.dumps(instance.metadata, ensure_ascii=False)
        
        with self.get_cursor() as cur:
            cur.execute("""
                INSERT OR REPLACE INTO multi_turn_instances 
                (hash_id, total_turn, turn_datas_json, metadata_json, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                instance.hash_id,
                instance.total_turn,
                turn_datas_json,
                metadata_json
            ))
    
    def get(self, hash_id: str) -> Optional[MultiTurnDataInstance]:
        """根据 hash_id 查询一条数据"""
        with self.get_cursor() as cur:
            cur.execute("SELECT * FROM multi_turn_instances WHERE hash_id = ?", (hash_id,))
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_instance(row)
    
    def update(self, instance: MultiTurnDataInstance) -> bool:
        """
        更新一个已存在的实例（必须存在）。
        如果传入的 instance 包含 mt_id，则不会更改 mt_id 字段。
        """
        if not self.exists(instance.hash_id):
            return False

        turn_datas_json = json.dumps([asdict(t) for t in instance.turn_datas], ensure_ascii=False)
        metadata_json = json.dumps(instance.metadata, ensure_ascii=False)

        # 检查是否有 mt_id 字段且不为 None
        if instance.mt_id is not None:
            with self.get_cursor() as cur:
                cur.execute("""
                    UPDATE multi_turn_instances
                    SET mt_id = ? ,total_turn = ?, turn_datas_json = ?, metadata_json = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE hash_id = ?
                """, (
                    instance.mt_id,
                    instance.total_turn,
                    turn_datas_json,
                    metadata_json,
                    instance.hash_id
                ))
                return cur.rowcount > 0

    def delete(self, hash_id: str) -> bool:
        """删除指定 hash_id 的数据"""
        with self.get_cursor() as cur:
            cur.execute("DELETE FROM multi_turn_instances WHERE hash_id = ?", (hash_id,))
            return cur.rowcount > 0

    def delete_by_turn(self, turn: int) -> int:
        """删除数据库中 total_turn > turn 的样本
        
        Args:
            turn: 回合数阈值，删除 total_turn 大于此值的样本
            
        Returns:
            删除的样本数量
        """
        with self.get_cursor() as cur:
            cur.execute("DELETE FROM multi_turn_instances WHERE total_turn > ?", (turn,))
            return cur.rowcount

    def exists(self, hash_id: str) -> bool:
        """检查 hash_id 是否已存在"""
        with self.get_cursor() as cur:
            cur.execute("SELECT 1 FROM multi_turn_instances WHERE hash_id = ? LIMIT 1", (hash_id,))
            return cur.fetchone() is not None

    def get_all_hashids(self) -> Set[str]:
        """返回所有已存在的 hash_id 集合"""
        with self.get_cursor() as cur:
            cur.execute("SELECT hash_id FROM multi_turn_instances")
            return {row["hash_id"] for row in cur.fetchall()}

    def get_incomplete_instances(self, source: str = "bigcodebench") -> List[MultiTurnDataInstance]:
        """
        返回所有 solution 或 test 不完整的 MultiTurnDataInstance 对象列表

        不完整的定义：turn_datas_json 中至少存在一个 PerTurnDataInstance，
        其 solution 或 test 字段为空字符串或 None

        Args:
            source: 指定数据来源，默认为 "bigcodebench"
        """
        incomplete_instances = []
        
        with self.get_cursor() as cur:
            cur.execute("SELECT * FROM multi_turn_instances")
            for row in cur.fetchall():
                try:
                    # 注意：sqlite3.Row 没有 get 方法，需用 row["metadata_json"]
                    metadata_json = row["metadata_json"] if "metadata_json" in row.keys() else "{}"
                    metadata = json.loads(metadata_json)
                    if metadata.get("source") != source:
                        continue
                        
                    turn_datas = json.loads(row["turn_datas_json"])
                    # 检查是否有任何一个 turn 的 solution 或 test 为空
                    is_incomplete = False
                    for turn_data in turn_datas:
                        solution = turn_data.get("solution", "")
                        test = turn_data.get("test", "")
                        # 兼容 None 的情况，避免 .strip() 报错
                        if not (isinstance(solution, str) and solution.strip()):
                            is_incomplete = True
                            break
                        if not (isinstance(test, str) and test.strip()):
                            is_incomplete = True
                            break
                    
                    if is_incomplete:
                        # 构造完整的 MultiTurnDataInstance 对象
                        instance = self._row_to_instance(row)
                        incomplete_instances.append(instance)
                        
                except (json.JSONDecodeError, KeyError, TypeError):
                    # 如果 JSON 解析失败或数据结构异常，也认为是不完整的
                    # 但只有在 source 匹配的情况下才添加
                    try:
                        metadata_json = row["metadata_json"] if "metadata_json" in row.keys() else "{}"
                        metadata = json.loads(metadata_json)
                        if metadata.get("source") == source:
                            instance = self._row_to_instance(row)
                            incomplete_instances.append(instance)
                    except (json.JSONDecodeError, KeyError, TypeError):
                        # 如果 metadata 也解析失败，跳过这个样本
                        continue
        
        return incomplete_instances

    def list_all(self) -> List[MultiTurnDataInstance]:
        """获取所有 MultiTurnDataInstance 实例"""
        with self.get_cursor() as cur:
            cur.execute("SELECT * FROM multi_turn_instances ORDER BY mt_id")
            return [self._row_to_instance(row) for row in cur.fetchall()]
    
    def list_all_by_source(self, source: str) -> List[MultiTurnDataInstance]:
        """根据 metadata 中的 source 字段获取 MultiTurnDataInstance 实例"""
        with self.get_cursor() as cur:
            # 使用 JSON_EXTRACT 函数从 metadata_json 中提取 source 字段
            cur.execute("""
                SELECT * FROM multi_turn_instances 
                WHERE JSON_EXTRACT(metadata_json, '$.source') = ? 
                ORDER BY mt_id
            """, (source,))
            return [self._row_to_instance(row) for row in cur.fetchall()]
    # =========================
    # 导入导出功能
    # =========================

    def setup_with_jsonl(self, jsonl_path: Path) -> None:
        """
        从 JSONL 文件初始化数据库：清空旧数据，导入新数据
        每行是一个 MultiTurnDataInstance 的 dict 结构
        """
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        # 清空现有数据
        with self.get_cursor() as cur:
            cur.execute("DELETE FROM multi_turn_instances")

        # 逐行导入
        instances = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    instance = self._dict_to_instance(data)
                    instances.append(instance)
                except Exception as e:
                    raise ValueError(f"Invalid JSONL format at line {line_num}: {e}")

        # 批量插入
        for instance in instances:
            self.add(instance)

    def export_jsonl(self, output_path: Path, source:str = "bigcodebench") -> None:
        """
        将数据库中所有数据导出为 JSONL 文件
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        instances = self.list_all()
        with open(output_path, 'w', encoding='utf-8') as f:
            for instance in instances:
                if instance.metadata["source"] != source:
                    continue
                f.write(json.dumps(asdict(instance), ensure_ascii=False) + '\n')

    def export_jsonl_without_solution_and_test(self, output_path: Path) -> None:
        """
        将数据库中数据导出为 JSONL，但将每一轮的 solution 和 test 置为空字符串
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        instances = self.list_all()
        with open(output_path, 'w', encoding='utf-8') as f:
            for instance in instances:
                instance_dict = asdict(instance)
                for turn in instance_dict.get("turn_datas", []):
                    turn["solution"] = ""
                    turn["test"] = ""
                f.write(json.dumps(instance_dict, ensure_ascii=False) + '\n')

    # =========================
    # 内部辅助方法
    # =========================

    def _row_to_instance(self, row: sqlite3.Row) -> MultiTurnDataInstance:
        """将数据库行转换为 MultiTurnDataInstance 对象"""
        turn_datas = [
            PerTurnDataInstance(**t) for t in json.loads(row["turn_datas_json"])
        ]
        metadata = json.loads(row["metadata_json"])
        return MultiTurnDataInstance(
            hash_id=row["hash_id"],
            mt_id=row["mt_id"],
            total_turn=row["total_turn"],
            turn_datas=turn_datas,
            metadata=metadata
        )

    def _dict_to_instance(self, data: Dict[str, Any]) -> MultiTurnDataInstance:
        """将 dict 转换为 MultiTurnDataInstance"""
        turn_datas = [PerTurnDataInstance(**t) for t in data["turn_datas"]]
        metadata = data.get("metadata", {})
        return MultiTurnDataInstance(
            hash_id=data["hash_id"],
            mt_id=data["mt_id"],
            total_turn=data["total_turn"],
            turn_datas=turn_datas,
            metadata=metadata
        )

    # =========================
    # 工具方法
    # =========================

    def count(self) -> int:
        """返回当前数据总量"""
        with self.get_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM multi_turn_instances")
            return cur.fetchone()[0]

    def clear(self) -> None:
        """清空所有数据"""
        with self.get_cursor() as cur:
            cur.execute("DELETE FROM multi_turn_instances")

    def close(self):
        """关闭资源（当前无长期连接）"""
        pass
