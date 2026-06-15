"""
============================================================
笔记历史存储模块 (notes_store.py)
============================================================
职责：
  1. 持久化存储生成的笔记（JSON 文件）
  2. 提供笔记的 CRUD 操作（列表 / 详情 / 删除）
  3. 支持按时间排序、按风格筛选

存储方式：
  - 使用 JSON 文件（backend/knowledge/notes_history.json）
  - 每条笔记包含完整的生成结果（query / style / context / framework / note / created_at）
  - 自动去重（相同 query+style 的笔记覆盖旧记录）

约定：
  - 所有操作线程安全（通过文件锁保护）
  - 空/损坏文件自动修复
============================================================
"""

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import KNOWLEDGE_DIR

logger = logging.getLogger(__name__)

# 笔记历史文件路径
NOTES_FILE = KNOWLEDGE_DIR / "notes_history.json"

# 最大保留条数（超过后自动清理最旧的）
MAX_NOTES = 100

# 线程锁（保证并发写入安全）
_lock = threading.Lock()


# ============================================================
# 内部工具函数
# ============================================================


def _load_notes() -> List[Dict[str, Any]]:
    """
    从 JSON 文件加载笔记列表。

    出参：List[Dict] — 笔记列表；文件不存在或损坏时返回空列表
    """
    if not NOTES_FILE.exists():
        return []

    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        logger.warning("笔记文件格式异常（非列表），重置为空")
        return []
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("笔记文件读取失败：%s，重置为空", exc)
        return []


def _save_notes(notes: List[Dict[str, Any]]) -> None:
    """
    将笔记列表写入 JSON 文件（原子写入：先写临时文件再重命名）。

    入参：
        notes : List[Dict] — 笔记列表
    """
    try:
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = NOTES_FILE.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)
        # 原子重命名（Windows 需要先删除目标文件）
        if os.name == "nt" and NOTES_FILE.exists():
            NOTES_FILE.unlink()
        tmp_path.rename(NOTES_FILE)
    except OSError as exc:
        logger.exception("笔记文件写入失败：%s", exc)
        raise


# ============================================================
# 公开 API
# ============================================================


def save_note(
    query: str,
    style: str,
    context: str,
    framework: str,
    note: str,
) -> Dict[str, Any]:
    """
    保存一条生成的笔记到历史记录。

    入参：
        query     : str — 用户查询
        style     : str — 笔记风格
        context   : str — 原文素材
        framework : str — 知识框架
        note      : str — 最终笔记

    出参：
        Dict — 保存后的笔记对象（含 id 和 created_at）

    说明：
        - 相同 query+style 的笔记会覆盖旧记录（更新 created_at）
        - 自动清理超过 MAX_NOTES 的最旧记录
    """
    with _lock:
        notes = _load_notes()

        # 查找是否已存在相同 query+style 的笔记
        existing_idx = None
        for i, n in enumerate(notes):
            if n.get("query") == query and n.get("style") == style:
                existing_idx = i
                break

        now_utc = datetime.now(timezone.utc)
        created_at = now_utc.strftime("%Y-%m-%d %H:%M:%S")

        entry = {
            "id": uuid.uuid4().hex[:12],
            "query": query,
            "style": style,
            "context": context,
            "framework": framework,
            "note": note,
            "created_at": created_at,
        }

        if existing_idx is not None:
            # 覆盖旧记录，保留原 ID
            entry["id"] = notes[existing_idx].get("id", entry["id"])
            notes[existing_idx] = entry
            logger.info("笔记已更新 | id=%s | query=%.60s", entry["id"], query)
        else:
            notes.insert(0, entry)  # 最新笔记放在最前面
            logger.info("笔记已保存 | id=%s | query=%.60s", entry["id"], query)

        # 清理超量旧记录
        if len(notes) > MAX_NOTES:
            removed = notes[MAX_NOTES:]
            notes = notes[:MAX_NOTES]
            logger.info("清理旧笔记 | 删除=%d 条", len(removed))

        _save_notes(notes)
        return entry


def list_notes(
    style: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """
    获取笔记列表（仅返回摘要信息，不含完整内容）。

    入参：
        style  : str | None — 按风格筛选（可选）
        limit  : int       — 返回条数上限
        offset : int       — 偏移量

    出参：
        Dict — {
            "total": 10,
            "notes": [
                {
                    "id": "abc123",
                    "query": "什么是RAG？",
                    "style": "outline",
                    "created_at": "2025-06-15 14:30:00",
                    "note_preview": "一、RAG概述\n..."
                }
            ]
        }
    """
    notes = _load_notes()

    # 按风格筛选
    if style and style in ("outline", "mindmap", "exam_points"):
        notes = [n for n in notes if n.get("style") == style]

    total = len(notes)

    # 分页
    notes_page = notes[offset : offset + limit]

    # 构建摘要（截取 note 前 150 字符作为预览）
    result = []
    for n in notes_page:
        note_text = n.get("note", "")
        preview = note_text[:150].replace("\n", " ")
        if len(note_text) > 150:
            preview += "…"
        result.append({
            "id": n.get("id", ""),
            "query": n.get("query", "")[:100],
            "style": n.get("style", "outline"),
            "created_at": n.get("created_at", ""),
            "note_preview": preview,
        })

    return {"total": total, "notes": result}


def get_note(note_id: str) -> Optional[Dict[str, Any]]:
    """
    获取单条笔记的完整内容。

    入参：
        note_id : str — 笔记 ID

    出参：
        Dict | None — 完整笔记对象；未找到返回 None
    """
    notes = _load_notes()
    for n in notes:
        if n.get("id") == note_id:
            return n
    return None


def delete_note(note_id: str) -> bool:
    """
    删除指定 ID 的笔记。

    入参：
        note_id : str — 笔记 ID

    出参：
        bool — 是否成功删除（未找到返回 False）
    """
    with _lock:
        notes = _load_notes()
        original_len = len(notes)
        notes = [n for n in notes if n.get("id") != note_id]
        if len(notes) == original_len:
            logger.debug("未找到笔记 id=%s，跳过删除", note_id)
            return False
        _save_notes(notes)
        logger.info("笔记已删除 | id=%s", note_id)
        return True


def delete_all_notes() -> int:
    """
    清空所有笔记历史。

    出参：
        int — 删除的笔记数量
    """
    with _lock:
        notes = _load_notes()
        count = len(notes)
        _save_notes([])
        logger.info("所有笔记已清空 | 删除=%d 条", count)
        return count
