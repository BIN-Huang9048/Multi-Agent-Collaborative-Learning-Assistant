"""
============================================================
向量库管理模块 (vector_store.py)
============================================================
职责：
  1. 管理 ChromaDB 向量库的完整生命周期（初始化 / 入库 / 检索 / 清空）
  2. 封装 LangChain Embedding 层，使用 DeepSeek Embedding（OpenAI 兼容协议）
  3. 实现严格的增量更新逻辑 —— 新文档仅追加向量，绝不重建全量索引
  4. 每条向量携带来源元数据（文件名、分片序号），检索结果原文返回

核心原则：
  - 增量更新：利用 ChromaDB 原生 `add` 追加语义，不对已有数据做 DELETE + INSERT
  - 本地持久化：所有数据写入 config.CHROMA_DIR，重启不丢失
  - 元数据溯源：检索结果保留 source / chunk_index，便于上游引用

依赖：
  - chromadb                              : 底层向量数据库
  - langchain_community.vectorstores.Chroma : LangChain Chroma 封装
  - langchain_openai.OpenAIEmbeddings     : OpenAI 兼容 Embedding（主力）
  - sentence-transformers (可选)          : 全离线本地 Embedding

约定：
  - 所有向量库读写操作均捕获异常并记录日志，不静默失败
  - 检索结果不做任何内容篡改，原文输出
  - 分片 ID 采用「批次UUID_源文件名_序号」保证全局唯一
============================================================
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# ----------------------------------------------------------
# 项目内部模块
# ----------------------------------------------------------
from app.config import (
    CHROMA_DIR,              # ChromaDB 持久化路径
    VECTOR_COLLECTION_NAME,  # 默认集合名称："knowledge"
    VECTOR_SEARCH_TOP_K,     # 默认检索 Top-K：5
    DEEPSEEK_API_KEY,        # DeepSeek API 密钥
    DEEPSEEK_API_BASE,       # DeepSeek API 基础 URL
    DEEPSEEK_EMBED_MODEL,    # DeepSeek Embedding 模型名称
)

# ----------------------------------------------------------
# 模块级日志
# ----------------------------------------------------------
logger = logging.getLogger(__name__)

# ============================================================
# 第三方库延迟加载（按实际调用时检测可用性）
# ============================================================

# ---- chromadb ----
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    _HAS_CHROMADB = True
except ImportError:  # pragma: no cover
    chromadb = None         # type: ignore
    ChromaSettings = None   # type: ignore
    _HAS_CHROMADB = False

# ---- langchain Chroma wrapper ----
try:
    from langchain_community.vectorstores import Chroma as LangChainChroma

    _HAS_LC_CHROMA = True
except ImportError:
    try:
        from langchain.vectorstores import Chroma as LangChainChroma

        _HAS_LC_CHROMA = True
    except ImportError:  # pragma: no cover
        LangChainChroma = None  # type: ignore
        _HAS_LC_CHROMA = False

# ---- OpenAI 兼容 Embedding ----
try:
    from langchain_openai import OpenAIEmbeddings

    _HAS_OPENAI_EMBED = True
except ImportError:
    try:
        from langchain.embeddings.openai import OpenAIEmbeddings

        _HAS_OPENAI_EMBED = True
    except ImportError:  # pragma: no cover
        OpenAIEmbeddings = None  # type: ignore
        _HAS_OPENAI_EMBED = False

# ---- HuggingFace 本地 Embedding（可选，离线场景）----
try:
    from langchain_huggingface import HuggingFaceEmbeddings

    _HAS_HF_EMBED = True
except ImportError:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings

        _HAS_HF_EMBED = True
    except ImportError:
        HuggingFaceEmbeddings = None  # type: ignore
        _HAS_HF_EMBED = False


# ============================================================
# Embedding 后端类型常量
# ============================================================

EMBED_BACKEND_OPENAI = "openai"        # OpenAI 兼容 API（DeepSeek，默认）
EMBED_BACKEND_HF = "huggingface"       # 本地 sentence-transformers 模型
EMBED_BACKEND_DUMMY = "dummy"          # 哑后端（仅测试用，返回零向量）

# DeepSeek Embedding 模型（通过 OpenAIEmbeddings 兼容接口调用）
_DEFAULT_DEEPSEEK_EMBED_MODEL = "deepseek-embed"

# HuggingFace 默认 Embedding 模型（轻量、中文友好）
_DEFAULT_HF_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


class VectorStoreManager:
    """
    ChromaDB 向量库管理器

    封装 ChromaDB 的完整操作接口，对上层（Agent / API）屏蔽底层细节。

    核心能力：
      - 增量入库：add_documents() 仅追加，不重建索引
      - 语义检索：retrieve_similar() 返回原文 + 来源元数据
      - 库清空：  clear_vector_store() 删除集合并重建空库

    使用示例：
        store = VectorStoreManager()
        store.add_documents(chunks, source_name="笔记.pdf")
        results = store.retrieve_similar("什么是 RAG", top_k=3)
        # results → [{"content": "...", "source": "笔记.pdf", "score": 0.12}, ...]
    """

    # ============================================================
    # 构造方法
    # ============================================================

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedding_backend: str = EMBED_BACKEND_OPENAI,
        embedding_model: Optional[str] = None,
    ):
        """
        初始化向量库管理器。

        入参：
            persist_dir       : Path | None
                ChromaDB 持久化目录，默认使用 config.CHROMA_DIR
            collection_name   : str | None
                集合名称，默认使用 config.VECTOR_COLLECTION_NAME
            embedding_backend : str
                Embedding 后端类型，可选值：
                - "openai"      → OpenAI 兼容 API / DeepSeek（需配置 DEEPSEEK_API_KEY）
                - "huggingface" → 本地 sentence-transformers（需安装该库）
                - "dummy"       → 哑后端（仅调试用，返回固定维度零向量）
            embedding_model   : str | None
                Embedding 模型名称，None 时使用各后端默认模型

        抛出：
            ImportError — 所选后端的依赖未安装
            ValueError  — 未知的 embedding_backend 值
        """
        # --- 路径与集合名 ---
        # 兼容 str 和 Path 输入，统一转为 Path 对象
        if persist_dir is not None:
            self.persist_dir: Path = Path(persist_dir)
        else:
            self.persist_dir: Path = CHROMA_DIR

        self.collection_name: str = (
            collection_name if collection_name is not None else VECTOR_COLLECTION_NAME
        )

        # --- Embedding 配置 ---
        self.embedding_backend: str = embedding_backend
        self.embedding_model: Optional[str] = embedding_model
        self._embedding_function: Any = None  # LangChain Embedding 实例

        # --- 底层实例（延迟初始化）---
        self._vectorstore: Any = None  # LangChain Chroma 封装实例

        # --- 确保知识目录存在 ---
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # --- 启动初始化（Embedding）---
        self._init_embedding_function()

        # --- 启动初始化（ChromaDB 向量库）---
        # 依赖缺失时不阻断构造，推迟到实际调用方法时再报错。
        # 这允许在开发/测试环境（未安装 chromadb）中先实例化再按需检查。
        try:
            self._init_vectorstore()
        except (ImportError, RuntimeError) as exc:
            logger.warning(
                "ChromaDB 向量库初始化失败（依赖可能未安装），"
                "后续 add_documents / retrieve_similar 将不可用。"
                "错误详情：%s",
                exc,
            )
            self._vectorstore = None

        # --- 启动诊断 ---
        collection_size = self.count()
        logger.info(
            "VectorStoreManager 初始化完成 | 持久化目录=%s | 集合=%s | "
            "Embedding后端=%s | 模型=%s | 已有向量数=%d",
            self.persist_dir,
            self.collection_name,
            self.embedding_backend,
            self.embedding_model or "(default)",
            collection_size,
        )

    # ============================================================
    # 方法 1：增量添加文档分片
    # ============================================================

    def add_documents(
        self,
        chunks: List[str],
        source_name: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        【方法1】增量添加文本分片到向量库，自动持久化。

        核心逻辑（严格增量）：
          - 为每个分片生成全局唯一 ID，通过 ChromaDB 的 `add` 追加写入
          - 不执行 DELETE、不重建集合、不修改已有向量
          - ChromaDB 底层自动将新向量持久化到磁盘

        入参：
            chunks         : List[str]
                已清洗 & 分片后的文本块列表（来自 FileHandler.split_text）
            source_name    : str
                来源文件名（如 "机器学习入门.pdf"），写入每条向量的元数据
            extra_metadata : Dict | None
                额外的元数据字段（如文件哈希、上传时间等），可选

        出参：
            int — 成功入库的分片数量

        抛出：
            ImportError — chromadb / langchain 未安装
            RuntimeError — 向量库写入失败
            ValueError   — chunks 为空列表

        说明：
            - 空分片（纯空白字符）会自动过滤，不计入入库数
            - 元数据中始终包含 source（文件名）和 chunk_index（分片序号）
        """
        # ---- 前置校验 ----
        if not chunks:
            logger.warning("add_documents: chunks 为空列表，跳过人库")
            return 0

        # 过滤空分片
        original_count = len(chunks)
        chunks = [c for c in chunks if c and c.strip()]
        filtered_count = original_count - len(chunks)
        if filtered_count > 0:
            logger.debug("过滤空分片 | 原始=%d | 有效=%d", original_count, len(chunks))

        if not chunks:
            logger.warning("add_documents: 过滤后无有效分片，跳过人库")
            return 0

        # ---- 依赖检查 ----
        self._ensure_ready()

        # ---- 生成唯一 ID ----
        # 使用「批次 UUID + 源文件名 + 序号」保证全局唯一
        # 即使同一文件被重复上传，每次也会生成新的 UUID 批次号
        batch_id: str = uuid.uuid4().hex[:12]
        # 清洗 source_name 中的特殊字符（避免 ChromaDB ID 解析异常）
        safe_source = self._sanitize_id_component(source_name)
        ids: List[str] = [
            f"{batch_id}_{safe_source}_{i:04d}"
            for i in range(len(chunks))
        ]

        # ---- 构造元数据 ----
        metadatas: List[Dict[str, Any]] = []
        for i in range(len(chunks)):
            meta: Dict[str, Any] = {
                "source": source_name,          # 来源文件名
                "chunk_index": i,               # 分片序号
                "batch_id": batch_id,           # 批次标识
            }
            if extra_metadata:
                meta.update(extra_metadata)
            metadatas.append(meta)

        # ---- 写入向量库 ----
        try:
            self._vectorstore.add_texts(
                texts=chunks,
                metadatas=metadatas,
                ids=ids,
            )
            # ChromaDB 默认自动持久化（duckdb+parquet 模式下写入即持久）
            # 无需显式调用 persist()
        except Exception as exc:
            logger.exception(
                "向量库增量写入失败 | 源文件=%s | 分片数=%d | 错误=%s",
                source_name,
                len(chunks),
                exc,
            )
            raise RuntimeError(
                f"向量库写入失败（源文件: {source_name}）：{exc}"
            ) from exc

        logger.info(
            "增量入库完成 | 源文件=%s | 入库分片=%d | 批次ID=%s",
            source_name,
            len(chunks),
            batch_id,
        )
        return len(chunks)

    # ============================================================
    # 方法 2：语义检索
    # ============================================================

    def retrieve_similar(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        【方法2】根据用户查询做语义检索，返回相似度最高的文档片段。

        入参：
            query           : str
                查询文本（自然语言问题或关键词）
            top_k           : int | None
                返回结果数量，默认使用 config.VECTOR_SEARCH_TOP_K（= 5）
            score_threshold : float | None
                相似度阈值，低于此值的结果会被过滤。
                注意：ChromaDB 返回的是 **距离** 而非相似度；
                距离越小越相关，默认无阈值。

        出参：
            List[Dict] — 按相似度降序排列的结果列表，每条结果包含：
                {
                    "content"     : str,   # 原始文档片段（不做任何篡改）
                    "source"      : str,   # 来源文件名
                    "chunk_index" : int,   # 分片序号（可追溯原文位置）
                    "distance"    : float, # ChromaDB 距离值（越小越相似）
                    "batch_id"    : str,   # 入库批次 ID
                }

        抛出：
            ImportError — chromadb / langchain 未安装
            RuntimeError — 检索过程异常

        说明：
            - 空集合返回空列表，不抛异常
            - 检索结果不做内容修改，完整保留原文上下文
        """
        if not query or not query.strip():
            logger.warning("retrieve_similar: 查询文本为空")
            return []

        k = top_k if top_k is not None else VECTOR_SEARCH_TOP_K

        # ---- 依赖检查 ----
        self._ensure_ready()

        # ---- 执行语义检索 ----
        try:
            # similarity_search_with_score 返回 (Document, score) 元组列表
            # ChromaDB 的 score 是距离值（L2 或 cosine 距离），越小越相似
            docs_with_scores = self._vectorstore.similarity_search_with_score(
                query,
                k=k,
            )
        except Exception as exc:
            logger.exception(
                "向量库检索失败 | query=%s | top_k=%d | 错误=%s",
                query[:100],
                k,
                exc,
            )
            raise RuntimeError(f"向量库检索失败：{exc}") from exc

        # ---- 格式化返回结果 ----
        results: List[Dict[str, Any]] = []
        for doc, score in docs_with_scores:
            result = {
                "content": doc.page_content,   # 原文片段，不做篡改
                "source": doc.metadata.get("source", "unknown"),
                "chunk_index": doc.metadata.get("chunk_index", -1),
                "distance": float(score),       # ChromaDB 距离值
                "batch_id": doc.metadata.get("batch_id", ""),
            }

            # ---- 可选：按距离阈值过滤 ----
            if score_threshold is not None and result["distance"] > score_threshold:
                logger.debug(
                    "检索结果被阈值过滤 | distance=%.4f > threshold=%.4f",
                    result["distance"],
                    score_threshold,
                )
                continue

            results.append(result)

        logger.info(
            "语义检索完成 | query=%.80s | 命中=%d条 | top_k=%d",
            query,
            len(results),
            k,
        )
        return results

    # ============================================================
    # 方法 3：清空向量库
    # ============================================================

    def clear_vector_store(self) -> int:
        """
        【方法3】清空当前集合中的所有向量数据。

        用途：
          - 开发调试 / 单元测试后的数据重置
          - 知识库全量更新前的清理（配合重新入库使用）

        入参：无

        出参：
            int — 删除前集合中的向量总数（用于确认清空范围）

        抛出：
            RuntimeError — 清空操作失败

        说明：
            - 删除后自动重建空集合，确保后续 add_documents 可直接使用
            - 持久化文件同步清理（ChromaDB 自动管理）
        """
        self._ensure_ready()

        # 记录删除前的向量数量
        deleted_count: int = self.count()

        try:
            # LangChain Chroma.delete_collection() 内部调用
            # chromadb.Collection.delete()，彻底删除集合及其持久化数据
            self._vectorstore.delete_collection()
            logger.info("向量库集合已删除 | 删除向量数=%d", deleted_count)
        except Exception as exc:
            logger.exception("向量库清空失败 | 错误=%s", exc)
            raise RuntimeError(f"向量库清空失败：{exc}") from exc

        # ---- 重建空集合 ----
        try:
            self._init_vectorstore()
            logger.info("向量库空集合已重建 | 集合名=%s", self.collection_name)
        except Exception as exc:
            logger.exception("向量库重建失败 | 错误=%s", exc)
            raise RuntimeError(
                f"向量库清空后重建失败，服务可能处于不可用状态：{exc}"
            ) from exc

        return deleted_count

    # ============================================================
    # 方法 4：按来源删除
    # ============================================================

    def delete_by_source(self, source_name: str) -> int:
        """
        【方法4】删除指定来源文件的所有向量数据。

        入参：
            source_name : str — 来源文件名（与入库时的 source 元数据匹配）

        出参：
            int — 删除的向量数量

        说明：
            - 若指定来源不存在，返回 0（不抛异常）
            - 底层通过 ChromaDB 的 metadata 过滤删除实现
        """
        self._ensure_ready()

        try:
            collection = self._vectorstore._collection
            if collection is None:
                logger.warning("向量库集合不可用，无法删除来源=%s", source_name)
                return 0

            # 先查询匹配的向量数量
            existing = collection.get(
                where={"source": source_name}
            )
            existing_ids = existing.get("ids", []) if existing else []
            count = len(existing_ids)

            if count == 0:
                logger.info("未找到来源=%s 的向量，跳过删除", source_name)
                return 0

            # 删除匹配的向量
            collection.delete(
                where={"source": source_name}
            )
            logger.info(
                "按来源删除完成 | 来源=%s | 删除向量数=%d",
                source_name,
                count,
            )
            return count

        except Exception as exc:
            logger.exception("按来源删除失败 | 来源=%s | 错误=%s", source_name, exc)
            raise RuntimeError(
                f"删除来源「{source_name}」的向量失败：{exc}"
            ) from exc

    # ============================================================
    # 辅助方法
    # ============================================================

    def count(self) -> int:
        """
        获取当前集合中的向量总数。

        出参：int — 向量数量；若集合不可用返回 0
        """
        if self._vectorstore is None:
            return 0
        try:
            # 通过底层 chromadb Collection 直接获取
            collection = self._vectorstore._collection
            return collection.count() if collection else 0
        except Exception as exc:
            logger.warning("获取向量数量失败：%s", exc)
            return 0

    def get_sources(self) -> List[str]:
        """
        获取集合中所有不重复的来源文件名。

        出参：List[str] — 去重后的来源文件名列表
        """
        if self._vectorstore is None:
            return []
        try:
            collection = self._vectorstore._collection
            if collection is None:
                return []
            result = collection.get()
            metadatas = result.get("metadatas", []) if result else []
            sources = set()
            for meta in metadatas:
                if meta and "source" in meta:
                    sources.add(meta["source"])
            return sorted(sources)
        except Exception as exc:
            logger.warning("获取来源列表失败：%s", exc)
            return []

    # ============================================================
    # 私有方法：Embedding 函数初始化
    # ============================================================

    def _init_embedding_function(self) -> None:
        """
        根据 embedding_backend 参数初始化 LangChain Embedding 实例。

        支持三种后端：
          1. openai      — OpenAI 兼容 API（使用 config 中的 key/base）
          2. huggingface — 本地 sentence-transformers（完全离线）
          3. dummy       — 固定维度零向量（仅用于 CI / 单元测试）
        """
        backend = self.embedding_backend.lower()

        # ============================================================
        # 后端 A：OpenAI 兼容（DeepSeek Embedding）
        # ============================================================
        if backend == EMBED_BACKEND_OPENAI:
            if not _HAS_OPENAI_EMBED:
                raise ImportError(
                    "DeepSeek Embedding 后端需要安装 langchain-openai，"
                    "请执行：pip install langchain-openai"
                )
            model = self.embedding_model or DEEPSEEK_EMBED_MODEL

            # 构造 OpenAIEmbeddings 参数（对接 DeepSeek 兼容端点）
            embed_kwargs: Dict[str, Any] = {
                "model": model,
                "openai_api_base": DEEPSEEK_API_BASE,  # DeepSeek Embedding 端点
            }

            # API Key：优先用环境变量 DEEPSEEK_API_KEY，否则用 config.DEEPSEEK_API_KEY
            api_key = os.getenv("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)
            if api_key and api_key != "your-deepseek-api-key":
                embed_kwargs["openai_api_key"] = api_key

            try:
                self._embedding_function = OpenAIEmbeddings(**embed_kwargs)
                logger.info(
                    "Embedding 初始化成功 | 后端=DeepSeek | 模型=%s | base=%s",
                    model,
                    embed_kwargs.get("openai_api_base", "default"),
                )
            except Exception as exc:
                logger.exception("DeepSeek Embedding 初始化失败")
                raise RuntimeError(f"DeepSeek Embedding 初始化失败：{exc}") from exc

        # ============================================================
        # 后端 B：HuggingFace 本地模型
        # ============================================================
        elif backend == EMBED_BACKEND_HF:
            if not _HAS_HF_EMBED:
                raise ImportError(
                    "HuggingFace Embedding 后端需要安装 langchain-huggingface "
                    "和 sentence-transformers，请执行：\n"
                    "pip install langchain-huggingface sentence-transformers"
                )
            model = self.embedding_model or _DEFAULT_HF_EMBED_MODEL

            # 离线模式：防止在 async FastAPI 上下文中发起同步 HTTP 请求失败
            # 模型应已缓存；若未缓存则让用户预先下载
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

            try:
                self._embedding_function = HuggingFaceEmbeddings(
                    model_name=model,
                    model_kwargs={"device": "cpu"},       # 默认 CPU，避免 CUDA 依赖
                    encode_kwargs={"normalize_embeddings": True},
                )
                logger.info(
                    "Embedding 初始化成功 | 后端=HuggingFace | 模型=%s | device=cpu",
                    model,
                )
            except Exception as exc:
                logger.exception("HuggingFace Embedding 初始化失败")
                raise RuntimeError(
                    f"HuggingFace Embedding 初始化失败：{exc}"
                ) from exc

        # ============================================================
        # 后端 C：哑后端（测试/CI）
        # ============================================================
        elif backend == EMBED_BACKEND_DUMMY:
            self._embedding_function = _DummyEmbeddings()
            logger.warning(
                "[WARNING] 使用哑 Embedding 后端 —— 所有向量为零向量，仅用于测试！"
            )

        # ============================================================
        # 未知后端
        # ============================================================
        else:
            raise ValueError(
                f"未知的 Embedding 后端「{backend}」，"
                f"可选值：{EMBED_BACKEND_OPENAI}, "
                f"{EMBED_BACKEND_HF}, {EMBED_BACKEND_DUMMY}"
            )

    def _init_vectorstore(self) -> None:
        """
        初始化 LangChain Chroma 向量库封装。

        行为：
          - 若持久化目录中已有集合数据，则自动加载（保留已有向量）
          - 若持久化目录为空，则创建新的空集合
          - 这是「增量更新」的基础：加载已有数据，追加新数据
        """
        if not _HAS_LC_CHROMA:
            raise ImportError(
                "LangChain Chroma 向量库封装不可用，请安装 langchain-community。\n"
                "执行：pip install langchain-community"
            )
        if not _HAS_CHROMADB:
            raise ImportError(
                "chromadb 库未安装。\n"
                "执行：pip install chromadb"
            )
        if self._embedding_function is None:
            raise RuntimeError(
                "Embedding 函数未初始化，无法创建向量库。"
            )

        try:
            self._vectorstore = LangChainChroma(
                collection_name=self.collection_name,
                embedding_function=self._embedding_function,
                persist_directory=str(self.persist_dir),
            )
            logger.debug(
                "Chroma 向量库就绪 | 集合=%s | 路径=%s",
                self.collection_name,
                self.persist_dir,
            )
        except Exception as exc:
            logger.exception("Chroma 向量库初始化失败")
            raise RuntimeError(f"Chroma 向量库初始化失败：{exc}") from exc

    def _ensure_ready(self) -> None:
        """
        确保向量库处于可用状态，若未初始化则抛异常。

        抛出：
            RuntimeError — 向量库未初始化
        """
        if self._vectorstore is None:
            raise RuntimeError(
                "向量库未初始化，请检查 chromadb / langchain 依赖是否安装，"
                "以及 Embedding 后端配置是否正确。"
            )

    @staticmethod
    def _sanitize_id_component(name: str) -> str:
        """
        清洗 ID 中的来源名称组件 —— 移除 ChromaDB ID 中不允许的特殊字符。

        入参：name : str — 原始来源名称
        出参：str — 安全的 ID 组件（仅保留字母、数字、下划线、连字符、中文）
        """
        import re

        # 去除扩展名
        name_no_ext = str(Path(name).stem) if "." in name else name
        # 替换非法字符为下划线
        safe = re.sub(r"[^0-9A-Za-z_一-鿿-]", "_", name_no_ext)
        # 去除连续下划线
        safe = re.sub(r"_{2,}", "_", safe)
        # 截断过长的名称（保留前 40 字符）
        return safe[:40] if len(safe) > 40 else safe


# ============================================================
# 哑 Embedding 后端（仅用于 CI / 单元测试）
# ============================================================


class _DummyEmbeddings:
    """
    哑 Embedding 实现 —— 对所有输入返回固定维度的零向量。

    用途：在 CI 环境或无 API 密钥时，保证代码逻辑可跑通。
    警告：零向量无任何语义区分能力，严禁用于生产环境！

    默认维度 512，与 BAAI/bge-small-zh-v1.5 保持一致，避免维度不匹配。
    """

    def __init__(self, dimension: int = 512):
        """
        入参：
            dimension : int — 向量维度，默认 512（对齐 bge-small-zh-v1.5）
        """
        self._dimension = dimension
        logger.warning(
            "_DummyEmbeddings 已激活 | 维度=%d | 所有向量将为零向量！",
            dimension,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量 Embedding（返回零向量列表）"""
        return [[0.0] * self._dimension for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """单条查询 Embedding（返回零向量）"""
        return [0.0] * self._dimension
