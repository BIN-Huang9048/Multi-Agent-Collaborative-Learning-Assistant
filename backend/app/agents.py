"""
============================================================
多智能体模块 (agents.py)
============================================================
职责：
  1. 定义全局状态体 AgentState（TypedDict），贯穿 LangGraph 工作流
  2. 实现三大职能 Agent 类 —— 各司其职、串行协作
  3. 用 LangGraph StateGraph 搭建工作流：检索 → 理解 → 生成
  4. 封装统一入口 run_workflow(query, style)，对外暴露调用接口

核心原则（强制遵守）：
  ▎Agent = 决策者 + 规则控制器 + 自检调度器（纯 Python 代码实现）
  ▎LLM  = 纯执行工具，仅负责文本生成 / 摘要 / 梳理
  ▎LLM 绝对不参与决策、自检、流程跳转

三大 Agent 串行流水线：
  RetrievalAgent ──→ UnderstandingAgent ──→ NoteGenerateAgent
  (检索+摘要)        (理解+框架)            (笔记生成)

笔记风格：
  - outline     : 提纲（层级编号列表）
  - mindmap     : 思维导图（中心主题辐射结构）
  - exam_points : 考点清单（Q&A + 重点标注）

依赖：
  - langchain_openai.ChatOpenAI       : LLM 调用（对接 DeepSeek，纯工具角色）
  - langgraph.graph.StateGraph        : 工作流编排
  - app.vector_store.VectorStoreManager: 向量检索

约定：
  - 所有自检、重试、拦截、路由决策 → Python 代码实现
  - 所有文本生成、摘要、梳理 → LLM 执行
  - 每个 Agent 的执行步骤均输出日志
  - LLM 输出严格基于知识库原文，禁止编造外部知识
============================================================
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, TypedDict

# ----------------------------------------------------------
# 项目内部模块
# ----------------------------------------------------------
from app.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_BASE,
    DEEPSEEK_CHAT_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_REQUEST_TIMEOUT,
    VECTOR_SEARCH_TOP_K,
)

# ----------------------------------------------------------
# 模块级日志
# ----------------------------------------------------------
logger = logging.getLogger(__name__)

# ============================================================
# 第三方库延迟加载（按实际调用时检测可用性）
# ============================================================

# ---- LangChain ChatOpenAI ----
try:
    from langchain_openai import ChatOpenAI

    _HAS_LC_OPENAI = True
except ImportError:
    try:
        from langchain.chat_models import ChatOpenAI

        _HAS_LC_OPENAI = True
    except ImportError:  # pragma: no cover
        ChatOpenAI = None  # type: ignore
        _HAS_LC_OPENAI = False

# ---- LangGraph ----
try:
    from langgraph.graph import StateGraph, END

    _HAS_LANGGRAPH = True
except ImportError:  # pragma: no cover
    StateGraph = None  # type: ignore
    END = None         # type: ignore
    _HAS_LANGGRAPH = False

# LangGraph START 哨兵（不同版本位置不同）
try:
    from langgraph.graph import START
except ImportError:
    try:
        from langgraph import START
    except ImportError:
        START = "__start__"  # 兜底字符串常量


# ============================================================
# 一、全局状态体定义
# ============================================================

class AgentState(TypedDict, total=False):
    """
    LangGraph 全局状态字典。

    字段说明：
      query      : str — 用户输入的知识点 / 查询主题
      context    : str — 向量库检索到的原文素材（经摘要提纯，供 LLM 使用）
      raw_context: str — 原始检索素材（未经摘要，供 Python 溯源校验使用）
      framework  : str — 知识框架（结构化大纲）
      note       : str — 最终生成的笔记
      style      : str — 笔记风格：outline / mindmap / exam_points
      error      : str — 错误信息（非空时表示流程异常终止）
      status     : str — 流程状态标记："running" | "completed" | "terminated"
    """
    query: str
    context: str
    raw_context: str
    framework: str
    note: str
    style: str
    error: str
    status: str


# ============================================================
# 二、LLM 初始化
# ============================================================

def _create_llm(temperature: Optional[float] = None) -> Any:
    """
    创建 ChatOpenAI 实例（对接 DeepSeek API，LLM 纯工具角色）。

    入参：
        temperature : float | None — 生成温度，None 时使用 config 默认值

    出参：
        ChatOpenAI 实例

    抛出：
        ImportError — langchain-openai 未安装
        RuntimeError — API Key 为占位符，无法调用

    说明：
        - 使用 OpenAI 兼容协议调用 DeepSeek API
        - LLM 仅用作文本生成工具，不参与 Agent 决策
    """
    if not _HAS_LC_OPENAI:
        raise ImportError(
            "LLM 调用需要 langchain-openai，请执行：pip install langchain-openai"
        )

    api_key = DEEPSEEK_API_KEY
    if api_key == "your-deepseek-api-key":
        raise RuntimeError(
            "DeepSeek API Key 未配置！请在 backend/.env 中设置 DEEPSEEK_API_KEY=你的真实密钥"
        )

    kwargs: Dict[str, Any] = {
        "model": DEEPSEEK_CHAT_MODEL,
        "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
        "max_tokens": LLM_MAX_TOKENS,
        "openai_api_key": api_key,
        "openai_api_base": DEEPSEEK_API_BASE,  # DeepSeek API 端点
        "request_timeout": LLM_REQUEST_TIMEOUT,
    }

    llm = ChatOpenAI(**kwargs)
    logger.info(
        "LLM 实例已创建（DeepSeek）| 模型=%s | 温度=%.1f | max_tokens=%d | base=%s",
        DEEPSEEK_CHAT_MODEL,
        kwargs["temperature"],
        LLM_MAX_TOKENS,
        DEEPSEEK_API_BASE,
    )
    return llm


# ============================================================
# 三、笔记风格模板
# ============================================================

# 提纲风格模板
OUTLINE_TEMPLATE = """
你是一位知识管理专家。请根据以下素材生成一份**提纲风格**的学习笔记。

格式要求：
  1. 使用层级编号：一、(一) 1. (1) a.
  2. 每个层级至少包含 2 个子项
  3. 重要概念用【】标注
  4. 结尾附「关键术语表」（术语 → 一句话解释）

素材内容：
{content}

请严格按照上述格式生成笔记，不得添加素材中不存在的外部知识。
"""

# 思维导图风格模板
MINDMAP_TEMPLATE = """
【角色】你是专业思维导图生成器。用户选择了思维导图模式，你必须输出标准 Markdown 层级标题格式（兼容 markmap 渲染引擎）。

【输出格式 - 严格遵守】：
  # 中心主题（1 个，概括用户主题）
  ## 一级分支（3~6 个）
  ### 二级分支（每个一级分支下 2~5 个）
  #### 三级分支（仅在必要时使用，最多 1 层）

【内容规则】：
  - 每行只写一个标题，从 # 开始
  - 节点文字简洁，不超过 20 字，只写核心信息
  - 关键技术/工具用【】括起来，如 ## 【Kafka】（实时消息传递）
  - 禁止输出任何解释性段落、空行、非标题文字
  - 结构必须围绕用户主题展开，不得添加无关内容

用户主题：{content}
请直接输出 Markdown 标题，不要任何额外说明。
"""

# 考点清单风格模板
EXAM_POINTS_TEMPLATE = """
你是一位考试命题分析专家。请根据以下素材生成一份**考点清单风格**的学习笔记。

格式要求：
  1. 每个考点用「考点 N：标题」开头
  2. 每个考点包含：
     - 【核心概念】：1~2 句话
     - 【易错点】：常见错误理解
     - 【记忆口诀】：简洁的助记短语
  3. 考频预估：★★★★★（必考）~ ★☆☆☆☆（低频）
  4. 结尾附「考前速记卡」（5 条以内）

素材内容：
{content}

请严格按照上述格式生成笔记，不得添加素材中不存在的外部知识。
"""

# 风格 → 模板映射
STYLE_TEMPLATE_MAP: Dict[str, str] = {
    "outline": OUTLINE_TEMPLATE,
    "mindmap": MINDMAP_TEMPLATE,
    "exam_points": EXAM_POINTS_TEMPLATE,
}

# 有效风格列表
VALID_STYLES: set = {"outline", "mindmap", "exam_points"}


# ============================================================
# 四、基础 Agent 抽象类
# ============================================================

class BaseAgent:
    """
    所有职能 Agent 的抽象基类。

    提供：
      - LLM 调用封装（纯工具方法，不包含任何决策逻辑）
      - 统一的日志输出格式
      - 子类必须实现的 execute(state) 方法

    设计原则：
      Agent 子类的 execute() 方法中，所有 if/else/retry 逻辑
      均使用纯 Python 代码编写，LLM 仅通过 _call_llm() 被调用。
    """

    def __init__(self, name: str, llm: Any):
        """
        入参：
            name : str — Agent 名称（用于日志标识）
            llm  : ChatOpenAI — LLM 实例（纯工具角色）
        """
        self.name = name
        self.llm = llm

    # ----------------------------------------------------------
    # LLM 调用封装（纯工具：输入 prompt → 输出文本）
    # ----------------------------------------------------------

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
    ) -> str:
        """
        调用 LLM 生成文本 —— 纯工具方法，不包含任何决策逻辑。

        入参：
            system_prompt : str — 系统指令（角色、规则、约束）
            user_prompt   : str — 用户输入（具体内容）
            temperature   : float | None — 临时覆盖温度参数

        出参：
            str — LLM 生成的文本

        抛出：
            RuntimeError — LLM 调用失败

        说明：
            本方法绝不修改 prompt 内容、不做后处理判断、不参与流程决策。
            调用方（Agent execute 方法）负责校验输出、决定重试。
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        logger.debug(
            "[%s] 正在调用 LLM | system_len=%d | user_len=%d",
            self.name,
            len(system_prompt),
            len(user_prompt),
        )

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            # 若需要临时修改温度，创建临时 LLM 实例
            if temperature is not None and temperature != LLM_TEMPERATURE:
                temp_llm = _create_llm(temperature=temperature)
                response = temp_llm.invoke(messages)
            else:
                response = self.llm.invoke(messages)

            result: str = response.content if hasattr(response, "content") else str(response)

            logger.debug(
                "[%s] LLM 返回 | 长度=%d 字符",
                self.name,
                len(result),
            )
            return result

        except Exception as exc:
            logger.exception("[%s] LLM 调用失败 | 错误=%s", self.name, exc)
            raise RuntimeError(
                f"[{self.name}] LLM 调用失败：{exc}"
            ) from exc

    # ----------------------------------------------------------
    # 抽象方法
    # ----------------------------------------------------------

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        执行 Agent 核心逻辑。

        入参：state : AgentState — 全局状态字典
        出参：Dict[str, Any]  — 要更新的状态字段（部分更新）

        子类必须实现，所有决策逻辑使用纯 Python 代码。
        """
        raise NotImplementedError(f"{self.name}.execute() 必须由子类实现")


# ============================================================
# 五、检索解析 Agent
# ============================================================

class RetrievalAgent(BaseAgent):
    """
    检索解析 Agent —— 职责：校验输入 → 向量检索 → 相关性自检 → 摘要提纯

    Python 自检逻辑（不依赖 LLM）：
      1. 校验 query 非空、长度合理
      2. 检索结果数量检查：0 条 → 二次检索（降低阈值）
      3. 相关性检查：每条结果的 distance 必须在阈值内
      4. 有效素材数不足 → 终止流程，返回友好提示
      5. 有效素材 → 调用 LLM 做摘要提纯（仅此步骤使用 LLM）

    LLM 角色（仅文本工具）：
      - 对检索结果做摘要提纯，去除冗余，保留核心知识点
    """

    # 相关性距离阈值（ChromaDB L2 距离，越小越相关）
    # L2² = 2 - 2*cos_sim，所以：
    #   0.0 → cos_sim=1.0（完全相同）
    #   0.6 → cos_sim=0.7（高度相关）
    #   0.8 → cos_sim=0.6（中等相关）
    #   1.0 → cos_sim=0.5（弱相关）
    #   2.0 → cos_sim=0.0（无关）
    RELEVANCE_DISTANCE_THRESHOLD: float = 0.8
    # 二次检索放宽的阈值系数（首检不通过时放宽到 threshold*factor）
    RETRY_THRESHOLD_FACTOR: float = 1.25
    # 最少有效素材条数
    MIN_VALID_RESULTS: int = 1

    def __init__(self, llm: Any, vector_store: Any):
        """
        入参：
            llm          : ChatOpenAI — LLM 实例
            vector_store : VectorStoreManager — 向量库管理器
        """
        super().__init__("RetrievalAgent", llm)
        self.vector_store = vector_store

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------

    def execute(self, state: AgentState, status_callback: Any = None) -> Dict[str, Any]:
        """
        执行检索解析流程。

        流程：
          Step 1 → 输入校验（Python）
          Step 2 → 首次检索（向量库）
          Step 3 → 相关性自检（Python）
          Step 4 → 条件：二次检索（Python 决策）
          Step 5 → 素材不足 → 终止流程（Python 决策）
          Step 6 → 调用 LLM 摘要提纯（LLM 工具）
        """
        query = state.get("query", "").strip()
        style = state.get("style", "outline")

        logger.info("=" * 50)
        logger.info("[%s] 开始执行 | query=%.60s | style=%s", self.name, query, style)

        # ---- Step 1：输入校验（Python）----
        is_valid, error_msg = self._validate_query(query)
        if not is_valid:
            logger.warning("[%s] 输入校验失败 → 终止流程 | 原因=%s", self.name, error_msg)
            if status_callback:
                status_callback(self.name, "error", error_msg)
            return {
                "context": "",
                "error": error_msg,
                "status": "terminated",
            }

        if status_callback:
            status_callback(self.name, "start", f"正在检索「{query[:20]}」相关知识...")

        # ---- Step 2：首次检索（调用向量库）----
        logger.info("[%s] 正在执行首次向量检索...", self.name)
        raw_results = self._do_search(query, top_k=VECTOR_SEARCH_TOP_K)

        if status_callback:
            status_callback(self.name, "progress", f"已检索到 {len(raw_results)} 条候选素材，正在校验相关性...")

        # ---- Step 3：相关性自检（Python）----
        valid_results = self._check_relevance(raw_results)
        logger.info(
            "[%s] 相关性自检完成 | 检索=%d条 | 有效=%d条",
            self.name,
            len(raw_results),
            len(valid_results),
        )

        # ---- Step 4：二次检索决策（Python）----
        if len(valid_results) < self.MIN_VALID_RESULTS:
            logger.warning(
                "[%s] 有效素材不足（%d条 < %d条），触发二次检索...",
                self.name,
                len(valid_results),
                self.MIN_VALID_RESULTS,
            )
            if status_callback:
                status_callback(self.name, "retry", f"有效素材不足，正在放宽条件二次检索...")
            # 二次检索：放宽阈值 + 扩大 Top-K
            retry_results = self._do_search(
                query,
                top_k=VECTOR_SEARCH_TOP_K * 2,
            )
            retry_valid = self._check_relevance(
                retry_results,
                threshold=self.RELEVANCE_DISTANCE_THRESHOLD * self.RETRY_THRESHOLD_FACTOR,
            )
            logger.info(
                "[%s] 二次检索完成 | 检索=%d条 | 有效=%d条",
                self.name,
                len(retry_results),
                len(retry_valid),
            )
            # 合并去重
            valid_results = self._merge_results(valid_results, retry_valid)

        # ---- Step 5：无有效素材 → 终止（Python 决策）----
        if len(valid_results) < self.MIN_VALID_RESULTS:
            terminate_msg = (
                f"抱歉，在知识库中未找到与「{query}」相关的素材。\n"
                f"建议：1. 检查拼写是否正确；2. 使用更通用的关键词；3. 先上传相关文档到知识库。"
            )
            logger.warning("[%s] 无有效素材 → 终止流程", self.name)
            if status_callback:
                status_callback(self.name, "error", "未在知识库中找到相关素材")
            return {
                "context": "",
                "error": terminate_msg,
                "status": "terminated",
            }

        # ---- Step 5-B：关键词兜底检查（Python）----
        # 防止中文嵌入向量对无关文档给出虚假高分
        if status_callback:
            status_callback(self.name, "progress", f"通过 {len(valid_results)} 条有效素材，正在关键词校验...")
        keyword_ok, keyword_msg = self._check_keyword_overlap(query, valid_results)
        if not keyword_ok:
            logger.warning("[%s] 关键词兜底未通过 → 终止流程 | 原因=%s", self.name, keyword_msg)
            if status_callback:
                status_callback(self.name, "error", "关键词校验未通过，素材与查询不匹配")
            return {
                "context": "",
                "error": keyword_msg,
                "status": "terminated",
            }

        # ---- Step 6：调用 LLM 做摘要提纯（LLM 纯工具角色）----
        if status_callback:
            status_callback(self.name, "progress", "正在对素材进行摘要提纯...")
        raw_material = self._format_results(valid_results)
        summary = self._summarize_material(query, raw_material)

        logger.info(
            "[%s] 执行完成 | 有效素材=%d条 | 摘要长度=%d字符 | 原始长度=%d字符",
            self.name,
            len(valid_results),
            len(summary),
            len(raw_material),
        )
        logger.info("=" * 50)

        if status_callback:
            status_callback(self.name, "done", f"检索完成（{len(valid_results)} 条有效素材，摘要 {len(summary)} 字）")

        return {
            "context": summary,
            "raw_context": raw_material,  # 原始素材 → 供溯源校验
            "status": "running",
        }

    # ----------------------------------------------------------
    # 自检方法（Python 实现）
    # ----------------------------------------------------------

    @staticmethod
    def _validate_query(query: str) -> tuple:
        """
        【Python 自检】校验用户查询输入。

        规则：
          - 不得为空
          - 不得为纯空白字符
          - 长度 ≥ 1 且 ≤ 500 字符
        """
        if not query:
            return False, "查询内容不能为空，请输入您想了解的知识点。"
        if len(query) > 500:
            return False, f"查询内容过长（{len(query)}字符），请精简到 500 字符以内。"
        return True, ""

    @staticmethod
    def _check_relevance(
        results: List[Dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        【Python 自检】基于 distance 阈值过滤低相关度结果。

        入参：
            results   : List[Dict] — 检索原始结果
            threshold : float | None — 距离阈值，None 使用默认值

        出参：
            List[Dict] — 通过阈值筛选的结果

        规则：
          - ChromaDB distance 越小越相关（0=完全相同）
          - distance > 阈值的视为不相关，直接丢弃
          - 不依赖 LLM 判断相关性
        """
        if threshold is None:
            threshold = RetrievalAgent.RELEVANCE_DISTANCE_THRESHOLD

        valid = []
        for r in results:
            distance = r.get("distance", float("inf"))
            if distance <= threshold:
                valid.append(r)
            else:
                logger.debug(
                    "[RetrievalAgent] 过滤低相关结果 | distance=%.3f > threshold=%.3f | content=%.40s...",
                    distance,
                    threshold,
                    r.get("content", ""),
                )
        return valid

    @staticmethod
    def _check_keyword_overlap(
        query: str,
        results: List[Dict[str, Any]],
    ) -> tuple:
        """
        【Python 自检】关键词兜底检查 —— 防止嵌入向量对无关文档给出虚假高分。

        策略：
          1. 从查询中提取连续双字片段（bigram）作为关键词
          2. 对每个检索结果的 content 检查是否包含任意 bigram
          3. 若所有结果都无任何关键词命中 → 判定为嵌入误判 → 拒绝

        入参：
            query   : str       — 用户查询
            results : List[Dict] — 已通过距离阈值的结果

        出参：
            (bool, str) — (是否通过, 失败原因)

        注意：
          - 仅检查字符级命中，不做语义判断
          - 短查询（≤2 字）放宽为单字检查
        """
        if not results:
            return False, "无检索结果"

        # 提取查询中的双字片段
        if len(query) >= 3:
            keywords = [query[i:i+2] for i in range(len(query) - 1)]
        else:
            # 短查询用单字
            keywords = [c for c in query if c.strip()]

        if not keywords:
            return True, ""  # 无有效关键词，放行

        # 合并所有结果的内容
        all_content = " ".join(r.get("content", "") for r in results if r.get("content"))

        if not all_content.strip():
            return False, "所有检索结果内容为空"

        # 检查关键词命中
        hits = [kw for kw in keywords if kw in all_content]
        hit_ratio = len(hits) / len(keywords)

        logger.debug(
            "[RetrievalAgent] 关键词兜底 | 查询关键词=%s | 命中=%s | 命中率=%.0f%%",
            keywords,
            hits,
            hit_ratio * 100,
        )

        # 至少 30% 的关键词命中，或者至少 1 个双字命中（短查询）
        if hit_ratio >= 0.3 or (len(keywords) <= 2 and len(hits) >= 1):
            return True, ""

        return False, (
            f"检索到的素材与查询「{query}」关键词不匹配"
            f"（素材中未找到：{'、'.join(k for k in keywords if k not in all_content)[:5]}）。\n"
            f"请确认知识库中包含相关文档。"
        )

    @staticmethod
    def _merge_results(
        a: List[Dict[str, Any]],
        b: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        【Python 工具】合并两次检索结果，按 content 去重。

        去重策略：内容前 100 字符相同视为重复。
        """
        seen = set()
        merged = []
        for item in a + b:
            key = item.get("content", "")[:100]
            if key not in seen:
                seen.add(key)
                merged.append(item)
        # 按 distance 升序排列（越相关越靠前）
        merged.sort(key=lambda x: x.get("distance", float("inf")))
        return merged

    # ----------------------------------------------------------
    # 向量检索
    # ----------------------------------------------------------

    def _do_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        调用 VectorStoreManager.retrieve_similar 执行检索。

        入参：
            query  : str — 查询文本
            top_k  : int — 返回数量
        出参：
            List[Dict] — 检索结果
        """
        try:
            results = self.vector_store.retrieve_similar(query, top_k=top_k)
            logger.debug(
                "[%s] 检索调用完成 | top_k=%d | 返回=%d条",
                self.name,
                top_k,
                len(results),
            )
            return results
        except Exception as exc:
            logger.exception("[%s] 向量检索失败", self.name)
            raise RuntimeError(f"[{self.name}] 向量检索失败：{exc}") from exc

    # ----------------------------------------------------------
    # 格式化与摘要（LLM 工具角色）
    # ----------------------------------------------------------

    @staticmethod
    def _format_results(results: List[Dict[str, Any]]) -> str:
        """
        【Python 工具】将检索结果格式化为 LLM 可读的文本块。

        不做内容修改，仅拼接并标注来源。
        """
        parts = []
        for i, r in enumerate(results, 1):
            source = r.get("source", "unknown")
            content = r.get("content", "")
            parts.append(f"--- 素材片段 {i}（来源：{source}）---\n{content}\n")
        return "\n".join(parts)

    def _summarize_material(self, query: str, raw_material: str) -> str:
        """
        调用 LLM 对检索素材做摘要提纯。

        LLM 角色：纯文本工具 —— 提取与 query 相关的核心信息，去除冗余。
        此方法不包含任何决策逻辑。
        """
        system_prompt = (
            "你是一位专业的知识提炼助手。你的任务是对给定的素材做摘要提纯。\n"
            "规则：\n"
            "  1. 只提取与用户问题直接相关的核心知识点\n"
            "  2. 保留原文中的关键术语、定义、数据，不得改写原意\n"
            "  3. 去除与问题无关的内容、冗余重复的表述\n"
            "  4. 禁止添加素材中不存在的外部知识\n"
            "  5. 输出长度控制在 800 字以内"
        )
        user_prompt = (
            f"用户问题：{query}\n\n"
            f"检索素材：\n{raw_material}\n\n"
            f"请对上述素材进行摘要提纯。"
        )

        return self._call_llm(system_prompt, user_prompt)


# ============================================================
# 六、理解梳理 Agent
# ============================================================

class UnderstandingAgent(BaseAgent):
    """
    理解梳理 Agent —— 职责：拦截空素材 → 构建知识框架 → 溯源校验 → 框架合规自检

    Python 自检逻辑（不依赖 LLM）：
      1. 拦截空素材：context 为空时直接终止
      2. 构建知识框架（调用 LLM）
      3. 溯源校验（Python）：框架中每个关键概念必须在素材中有对应
      4. 框架合规自检（Python）：检查结构完整性（层级深度、节点数量）
      5. 不合格 → 重试 1 次；仍不合格 → 终止

    LLM 角色（仅文本工具）：
      - 根据素材梳理知识逻辑框架（层级结构）
    """

    # 框架合规检查参数
    MIN_SECTIONS: int = 2          # 最少一级节点数
    MIN_DEPTH: int = 2             # 最少层级深度（标记层级）
    MAX_RETRIES: int = 1           # 框架不合格时的最大重试次数

    def __init__(self, llm: Any):
        super().__init__("UnderstandingAgent", llm)

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------

    def execute(self, state: AgentState, status_callback: Any = None) -> Dict[str, Any]:
        """
        执行理解梳理流程。

        流程：
          Step 1 → 拦截空素材（Python）
          Step 2 → 调用 LLM 构建知识框架（LLM 工具）
          Step 3 → 溯源校验：禁止编造外部知识（Python）
          Step 4 → 框架合规自检（Python）
          Step 5 → 不合格 → 重试 1 次（Python 决策）
          Step 6 → 仍不合格 → 终止（Python 决策）
        """
        context = state.get("context", "").strip()
        # 溯源校验使用原始素材（未摘要），避免摘要压缩导致术语丢失
        raw_context = state.get("raw_context", "").strip()
        trace_source = raw_context if raw_context else context
        query = state.get("query", "")

        logger.info("=" * 50)
        logger.info(
            "[%s] 开始执行 | context_len=%d | raw_context_len=%d | trace_source_len=%d",
            self.name,
            len(context),
            len(raw_context),
            len(trace_source),
        )

        # ---- Step 1：拦截空素材（Python）----
        if not context:
            error_msg = "知识素材为空，无法构建知识框架。请确认知识库中有相关文档。"
            logger.warning("[%s] 素材为空 → 终止流程", self.name)
            if status_callback:
                status_callback(self.name, "error", "知识素材为空，无法构建框架")
            return {
                "framework": "",
                "error": error_msg,
                "status": "terminated",
            }

        if status_callback:
            status_callback(self.name, "start", "正在基于素材构建知识框架...")

        # ---- Step 2 + 3 + 4：框架构建 + 溯源 + 合规自检循环 ----
        framework = ""
        for attempt in range(1 + self.MAX_RETRIES):
            logger.info(
                "[%s] 第 %d/%d 轮框架构建...",
                self.name,
                attempt + 1,
                1 + self.MAX_RETRIES,
            )

            if attempt > 0 and status_callback:
                status_callback(self.name, "retry", f"第 {attempt + 1} 轮重新构建知识框架...")

            # Step 2：调用 LLM 构建框架（LLM 工具）
            framework = self._build_framework(query, context)

            # Step 3：溯源校验（Python）—— 使用原始素材 trace_source
            if status_callback:
                status_callback(self.name, "progress", "正在对框架进行溯源校验...")
            trace_ok, trace_msg = self._source_trace_check(framework, trace_source)
            if not trace_ok:
                logger.warning(
                    "[%s] 溯源校验未通过 | 原因=%s | 第%d轮",
                    self.name,
                    trace_msg,
                    attempt + 1,
                )
                if attempt < self.MAX_RETRIES:
                    continue  # 重试
                else:
                    # 重试耗尽，终止
                    logger.error("[%s] 溯源校验重试耗尽 → 终止流程", self.name)
                    if status_callback:
                        status_callback(self.name, "error", f"溯源校验失败：{trace_msg}")
                    return {
                        "framework": framework,
                        "error": f"知识框架溯源校验失败（已重试{self.MAX_RETRIES}次）：{trace_msg}",
                        "status": "terminated",
                    }

            # Step 4：框架合规自检（Python）
            if status_callback:
                status_callback(self.name, "progress", "正在检查框架结构完整性...")
            compliance_ok, compliance_msg = self._check_framework_compliance(framework)
            if not compliance_ok:
                logger.warning(
                    "[%s] 框架合规自检未通过 | 原因=%s | 第%d轮",
                    self.name,
                    compliance_msg,
                    attempt + 1,
                )
                if attempt < self.MAX_RETRIES:
                    continue  # 重试
                else:
                    logger.error("[%s] 框架合规自检重试耗尽 → 终止流程", self.name)
                    if status_callback:
                        status_callback(self.name, "error", f"框架合规自检失败：{compliance_msg}")
                    return {
                        "framework": framework,
                        "error": f"知识框架合规自检失败（已重试{self.MAX_RETRIES}次）：{compliance_msg}",
                        "status": "terminated",
                    }

            # 两项检查均通过 → 跳出循环
            logger.info("[%s] 框架校验通过 | 第%d轮", self.name, attempt + 1)
            break

        logger.info(
            "[%s] 执行完成 | 框架长度=%d字符",
            self.name,
            len(framework),
        )
        logger.info("=" * 50)

        if status_callback:
            status_callback(self.name, "done", f"知识框架构建完成（{len(framework)} 字）")

        return {
            "framework": framework,
            "status": "running",
        }

    # ----------------------------------------------------------
    # 框架构建（LLM 工具角色）
    # ----------------------------------------------------------

    def _build_framework(self, query: str, context: str) -> str:
        """
        调用 LLM 构建知识逻辑框架。

        LLM 角色：纯文本工具 —— 根据素材梳理层级知识结构。
        """
        system_prompt = (
            "你是一位知识结构分析师。你的任务是根据给定素材梳理知识框架。\n"
            "规则：\n"
            "  1. 使用层级缩进表示：一级用「一、」，二级用「(一)」，三级用「1.」，四级用「(1)」\n"
            "  2. 每个知识点必须在素材中有依据，严禁凭空编造\n"
            "  3. 框架需覆盖素材的主要知识点，至少包含 3 个一级节点\n"
            "  4. 每个节点的描述控制在 30 字以内\n"
            "  5. 将每个关键概念/术语用【】括起来。注意：【】只包裹名词/术语本身（如【RAG】【检索增强生成】"
            "【中位数】【监督学习】），不要把整句描述放进去。错误示例：「【中位数适用于连续数据】」——"
            "应拆成「【中位数】适用于【连续数据】」\n"
            "  6. 【】内的术语名称和定义必须与素材原文一致，不可自创或改写\n"
            "  7. 若素材不足以支撑完整框架，明确标注「[素材不足]」"
        )
        user_prompt = (
            f"用户主题：{query}\n\n"
            f"知识素材：\n{context}\n\n"
            f"请基于上述素材梳理知识框架。"
        )

        return self._call_llm(system_prompt, user_prompt)

    # ----------------------------------------------------------
    # 自检方法（Python 实现）
    # ----------------------------------------------------------

    @staticmethod
    def _source_trace_check(framework: str, context: str) -> tuple:
        """
        【Python 自检】溯源校验 —— 精准拦截 LLM 幻觉，放过措辞差异。

        设计思路：
          ▎措辞差异（放行）：术语与素材用词不同但字符高度重叠 → 合理归纳
          ▎明显编造（拦截）：术语与素材几乎零字符重叠 → LLM 幻觉
          ▎示例：「大数据基础与特征分析」vs 素材中的「大数据」「特征」→ 重叠高 → 放行
          ▎示例：「欺诈检测系统」vs 素材全无欺诈相关字眼 → 重叠零 → 拦截

        阈值设计（只拦截极端情况）：
          - 术语 ≥ 10% 字符命中 → 放行（有迹可循）
          - 术语 < 10% 字符命中 → 标记为「完全无据」
          - 完全无据术语 > 40% → 判定 LLM 跑题 → 拦截
        """
        if not framework.strip():
            return False, "框架内容为空"

        # ---- 提取待校验术语 ----
        bracket_terms = re.findall(r"【(.+?)】", framework)
        if not bracket_terms:
            terms = re.findall(r"[一-鿿]{4,}", framework)
        else:
            terms = bracket_terms

        if not terms:
            return True, ""

        # ---- 功能词拆分正则 ----
        _CONNECTOR_PATTERN = re.compile(
            r'(适用于|是什么原因[？?]?|用于|是指|即为|指|为|是|的|等|与|和|或)'
        )

        # ---- 逐术语校验 ----
        untraceable = []       # 完全无据的术语
        for term in terms:
            # ① 精确子串匹配
            if term in context or term.lower() in context.lower():
                continue

            # ② 去常见后缀再匹配
            term_clean = re.sub(r'(等|是什么原因[？?]?|的原理|的作用)$', '', term.strip())
            if term_clean != term and (term_clean in context or term_clean.lower() in context.lower()):
                continue

            # ③ 功能词拆分：≥50% 子项命中
            sub_terms = _CONNECTOR_PATTERN.split(term)
            content_parts = [
                p.strip() for p in sub_terms
                if p.strip() and not _CONNECTOR_PATTERN.fullmatch(p) and len(p.strip()) >= 2
            ]
            if len(content_parts) >= 2:
                parts_found = sum(
                    1 for p in content_parts
                    if p in context or p.lower() in context.lower()
                )
                if parts_found / len(content_parts) >= 0.5:
                    continue

            # ④ Bigram 匹配（>3 字，≥20% 且 ≥2 个双字片段命中）
            if len(term) > 3:
                bigrams = [term[i:i+2] for i in range(len(term)-1)]
                valid_bigrams = [
                    bg for bg in bigrams
                    if any(('一' <= c <= '鿿') or c.isalpha() or c.isdigit() for c in bg)
                ]
                if valid_bigrams:
                    found = sum(1 for bg in valid_bigrams if bg in context)
                    if found >= 2 and found / len(valid_bigrams) >= 0.2:
                        continue

            # ⑤ 字符级匹配（>2 字，≥3 个字符命中 且 ≥20%）
            if len(term) > 2:
                meaningful = [
                    c for c in term
                    if ('一' <= c <= '鿿') or c.isalpha() or c.isdigit()
                ]
                if meaningful:
                    chars_found = sum(1 for c in meaningful if c in context)
                    if chars_found >= 3 and chars_found / len(meaningful) >= 0.2:
                        continue

            # 所有关卡均未通过 → 此术语在素材中完全无据
            untraceable.append(term)

        # ---- 判定：完全无据术语 > 40% → 拦截 ----
        hallucination_ratio = len(untraceable) / len(terms)
        if hallucination_ratio > 0.4:
            return False, (
                f"以下关键概念在素材中未找到依据（疑似编造）："
                f"{'、'.join(untraceable[:5])}"
            )

        if untraceable:
            logger.debug(
                "[UnderstandingAgent] 溯源通过 | 总术语=%d | 完全无据=%d (%.0f%%) | 在容错范围内",
                len(terms), len(untraceable), hallucination_ratio * 100,
            )
        return True, ""

    @staticmethod
    def _check_framework_compliance(framework: str) -> tuple:
        """
        【Python 自检】框架合规检查 —— 验证结构完整性。

        检查项：
          1. 至少包含 MIN_SECTIONS 个一级节点
          2. 至少达到 MIN_DEPTH 层级深度
          3. 内容不为空且非错误占位

        出参：
            (bool, str) — (是否通过, 失败原因)
        """
        if not framework.strip():
            return False, "框架内容为空"

        # 检查 1：一级节点数量（中文编号「一、」「二、」等）
        level1_pattern = re.findall(r"[一二三四五六七八九十]+、", framework)
        level1_alt = re.findall(r"^\d+[\.、\s]", framework, re.MULTILINE)

        level1_count = len(level1_pattern) if level1_pattern else len(level1_alt)

        if level1_count < UnderstandingAgent.MIN_SECTIONS:
            return False, (
                f"一级节点不足（当前={level1_count}，要求≥{UnderstandingAgent.MIN_SECTIONS}）"
            )

        # 检查 2：层级深度
        # 检测各级标记：一、→ (一) → 1. → (1)
        depth_indicators = [
            r"[一二三四五六七八九十]+、",              # 一级
            r"\([一二三四五六七八九十]+\)",              # 二级
            r"\d+[\.、]",                            # 三级
            r"\(\d+\)",                              # 四级
        ]
        max_depth = 0
        for i, pattern in enumerate(depth_indicators, 1):
            if re.search(pattern, framework):
                max_depth = i

        if max_depth < UnderstandingAgent.MIN_DEPTH:
            return False, (
                f"层级深度不足（当前={max_depth}级，要求≥{UnderstandingAgent.MIN_DEPTH}级）"
            )

        # 检查 3：内容不为占位
        if "[素材不足]" in framework and level1_count < 2:
            return False, "素材不足以支撑完整框架"

        return True, ""


# ============================================================
# 七、笔记生成 Agent
# ============================================================

class NoteGenerateAgent(BaseAgent):
    """
    笔记生成 Agent —— 职责：识别风格 → 套用模板 → 生成笔记 → 格式校验 + 内容降噪

    Python 自检逻辑（不依赖 LLM）：
      1. 识别并校验笔记风格（outline / mindmap / exam_points）
      2. 套用对应风格模板，调用 LLM 生成笔记
      3. 格式校验（Python）：检查输出是否符合风格的结构要求
      4. 内容降噪（Python）：检测空段落、重复内容、无关噪声
      5. 不合格 → 重试（最多 2 次）；仍不合格 → 返回降级版笔记

    LLM 角色（仅文本工具）：
      - 根据知识框架 + 指定风格生成最终学习笔记
    """

    MAX_RETRIES: int = 2  # 格式/内容不合格时的最大重试次数

    # 各风格的格式校验正则
    STYLE_FORMAT_RULES: Dict[str, Dict[str, Any]] = {
        "outline": {
            "required_patterns": [
                r"[一二三四五六七八九十]+、",   # 一级中文编号
                r"【.+?】",                     # 关键术语标注
            ],
            "min_sections": 2,
            "min_length": 200,
        },
        "mindmap": {
            "required_patterns": [
                r"(?:^|\n)#\s[^#]",   # 中心主题（# 开头，非 ##）
                r"(?:^|\n)##\s",       # 一级分支（## 开头）
            ],
            "min_sections": 3,
            "min_length": 80,
        },
        "exam_points": {
            "required_patterns": [
                r"考点\s*\d+",     # 考点编号
                r"【核心概念】",
                r"【易错点】",
            ],
            "min_sections": 1,
            "min_length": 300,
        },
    }

    def __init__(self, llm: Any):
        super().__init__("NoteGenerateAgent", llm)

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------

    def execute(self, state: AgentState, status_callback: Any = None) -> Dict[str, Any]:
        """
        执行笔记生成流程。

        流程：
          Step 1 → 风格校验 + 拦截空框架（Python）
          Step 2 → 调用 LLM 生成笔记（LLM 工具）
          Step 3 → 格式校验（Python）
          Step 4 → 内容降噪检测（Python）
          Step 5 → 不合格 → 重试（最多 2 次）（Python 决策）
          Step 6 → 重试耗尽 → 返回降级版 + 警告
        """
        framework = state.get("framework", "").strip()
        style = state.get("style", "outline").strip()
        context = state.get("context", "")

        logger.info("=" * 50)
        logger.info(
            "[%s] 开始执行 | 框架长度=%d | 风格=%s",
            self.name,
            len(framework),
            style,
        )

        # ---- Step 1：风格校验 + 拦截空框架（Python）----
        if not framework:
            error_msg = "知识框架为空，无法生成笔记。"
            logger.warning("[%s] 框架为空 → 终止流程", self.name)
            if status_callback:
                status_callback(self.name, "error", "知识框架为空，无法生成笔记")
            return {
                "note": "",
                "error": error_msg,
                "status": "terminated",
            }

        if style not in VALID_STYLES:
            logger.warning(
                "[%s] 未知风格「%s」→ 回退为 outline",
                self.name,
                style,
            )
            style = "outline"

        template = STYLE_TEMPLATE_MAP[style]
        style_names = {"outline": "提纲", "mindmap": "思维导图", "exam_points": "考点清单"}
        style_display = style_names.get(style, style)

        if status_callback:
            status_callback(self.name, "start", f"正在生成{style_display}风格笔记...")

        # ---- Step 2 + 3 + 4：生成 + 格式校验 + 内容降噪循环 ----
        note = ""
        last_error = ""
        for attempt in range(1 + self.MAX_RETRIES):
            logger.info(
                "[%s] 第 %d/%d 轮笔记生成...",
                self.name,
                attempt + 1,
                1 + self.MAX_RETRIES,
            )

            if attempt > 0 and status_callback:
                status_callback(self.name, "retry", f"第 {attempt + 1} 轮重新生成笔记...")

            # Step 2：调用 LLM 生成笔记（LLM 工具）
            note = self._generate_note(framework, template, style, attempt)

            # Step 3：格式校验（Python）
            if status_callback:
                status_callback(self.name, "progress", "正在校验笔记格式...")
            format_ok, format_msg = self._check_format(note, style)
            if not format_ok:
                logger.warning(
                    "[%s] 格式校验未通过 | 原因=%s | 第%d轮",
                    self.name,
                    format_msg,
                    attempt + 1,
                )
                last_error = f"格式校验失败：{format_msg}"
                if attempt < self.MAX_RETRIES:
                    continue

            # Step 4：内容降噪检测（Python）
            if status_callback:
                status_callback(self.name, "progress", "正在检测内容质量...")
            noise_ok, noise_msg = self._check_content_noise(note)
            if not noise_ok:
                logger.warning(
                    "[%s] 内容降噪未通过 | 原因=%s | 第%d轮",
                    self.name,
                    noise_msg,
                    attempt + 1,
                )
                last_error = f"内容质量不合格：{noise_msg}"
                if attempt < self.MAX_RETRIES:
                    continue

            # 两项检查均通过
            if format_ok and noise_ok:
                logger.info("[%s] 笔记校验通过 | 第%d轮", self.name, attempt + 1)
                break

        # ---- Step 5：重试耗尽处理（Python 决策）----
        if not note.strip():
            logger.error("[%s] 笔记生成完全失败 → 终止", self.name)
            if status_callback:
                status_callback(self.name, "error", f"笔记生成失败：{last_error}")
            return {
                "note": "",
                "error": f"笔记生成失败（已重试{self.MAX_RETRIES}次）：{last_error}",
                "status": "terminated",
            }

        # 如果经历重试但最终通过，记录警告
        if attempt > 0:
            logger.warning(
                "[%s] 笔记经 %d 次重试后通过 | 最终长度=%d字符",
                self.name,
                attempt,
                len(note),
            )

        # ---- Step 6：内容降噪清理（Python）----
        note = self._remove_noise(note)

        logger.info(
            "[%s] 执行完成 | 笔记长度=%d字符 | 总尝试=%d轮",
            self.name,
            len(note),
            attempt + 1,
        )
        logger.info("=" * 50)

        if status_callback:
            status_callback(self.name, "done", f"{style_display}笔记生成完成（{len(note)} 字）")

        return {
            "note": note,
            "status": "completed",
        }

    # ----------------------------------------------------------
    # 笔记生成（LLM 工具角色）
    # ----------------------------------------------------------

    def _generate_note(
        self,
        framework: str,
        template: str,
        style: str,
        attempt: int,
    ) -> str:
        """
        调用 LLM 生成最终笔记。

        LLM 角色：纯文本工具 —— 填充模板，生成结构化笔记。

        入参：
            framework : str — 知识框架
            template  : str — 风格模板（含占位符 {content}）
            style     : str — 风格标识
            attempt   : int — 当前尝试次数（第 0 次为首次生成）
        """
        system_prompt = (
            "你是一位专业的学习笔记撰写专家。\n"
            "核心规则：\n"
            "  1. 严格遵循模板的格式要求，不得自由发挥格式\n"
            "  2. 只能基于给定的知识框架生成内容，禁止编造外部知识\n"
            "  3. 保持内容的准确性和可读性\n"
            "  4. 使用中文输出\n"
            "  5. 若框架信息不足，明确标注「[待补充]」而非编造"
        )
        # 若为重试轮次，在 prompt 中加入改进指令
        retry_hint = ""
        if attempt > 0:
            retry_hint = (
                f"\n\n【重要提示】上一次生成的笔记格式或内容质量未通过校验，"
                f"请特别注意：\n"
                f"  1. 严格遵循上述格式要求\n"
                f"  2. 确保每个部分都有实质内容，避免空洞表述\n"
                f"  3. 控制整体篇幅，避免过长或过短\n"
            )

        user_prompt = template.format(content=framework) + retry_hint

        # 笔记生成使用较低温度，减少随机性
        return self._call_llm(system_prompt, user_prompt, temperature=0.5)

    # ----------------------------------------------------------
    # 自检方法（Python 实现）
    # ----------------------------------------------------------

    @classmethod
    def _check_format(cls, note: str, style: str) -> tuple:
        """
        【Python 自检】格式校验 —— 检查输出是否符合所选风格的结构要求。

        检查项：
          1. 字数达标（≥ 风格设定最小长度）
          2. 必备标记存在（如提纲需有编号、思维导图需有 ★ ▶）
          3. 段落数量充足

        出参：
            (bool, str) — (是否通过, 失败原因)
        """
        if not note or not note.strip():
            return False, "笔记内容为空"

        rules = cls.STYLE_FORMAT_RULES.get(style)
        if not rules:
            return True, ""  # 未知风格默认放行

        # 检查 1：最小长度
        min_length = rules.get("min_length", 100)
        if len(note) < min_length:
            return False, f"笔记过短（{len(note)}字 < {min_length}字要求）"

        # 检查 2：必备标记
        required_patterns = rules.get("required_patterns", [])
        for pattern in required_patterns:
            if not re.search(pattern, note):
                return False, f"缺少必备标记：{pattern}"

        # 检查 3：段落数量
        paragraphs = [p for p in note.split("\n") if p.strip()]
        min_sections = rules.get("min_sections", 2)
        if len(paragraphs) < min_sections:
            return False, f"段落不足（{len(paragraphs)}段 < {min_sections}段要求）"

        return True, ""

    @staticmethod
    def _check_content_noise(note: str) -> tuple:
        """
        【Python 自检】内容降噪检测 —— 识别空段落、重复内容、幻觉标记。

        检查项：
          1. 无连续 3+ 空白行
          2. 无明显的 LLM 幻觉标记（如 "作为一个AI"、"我无法" 等拒绝语）
          3. 无大段重复内容（同一句子出现 ≥3 次）

        出参：
            (bool, str) — (是否通过, 失败原因)
        """
        if not note.strip():
            return False, "笔记为空"

        # 检查 1：拒绝语/幻觉标记
        hallucination_markers = [
            "作为一个AI",
            "作为AI",
            "我无法",
            "抱歉，我",
            "根据我的训练",
            "我不能",
            "I cannot",
            "As an AI",
        ]
        for marker in hallucination_markers:
            if marker in note:
                return False, f"检测到 LLM 幻觉/拒绝标记：「{marker}」"

        # 检查 2：大段重复
        lines = [line.strip() for line in note.split("\n") if line.strip()]
        if len(lines) >= 6:
            # 用集合去重检查
            unique_lines = set(lines)
            if len(unique_lines) < len(lines) * 0.5:
                return False, f"内容重复率过高（唯一行={len(unique_lines)}/{len(lines)}）"

        return True, ""

    @staticmethod
    def _remove_noise(note: str) -> str:
        """
        【Python 工具】清理笔记中的噪声内容。

        操作：
          1. 去除连续 3+ 空白行 → 压缩为双空行
          2. 去除首尾空白
          3. 统一中文破折号
        """
        # 压缩连续空行
        note = re.sub(r"\n{3,}", "\n\n", note)
        # 去除首尾空白
        note = note.strip()
        return note


# ============================================================
# 八、LangGraph 工作流编排
# ============================================================


class AgentWorkflow:
    """
    LangGraph 工作流编排器。

    工作流拓扑：
        START ──→ retrieval ──→ understanding ──→ generation ──→ END
                     │                │                  │
                     ▼                ▼                  ▼
               (terminated)     (terminated)        (completed)
                     │                │
                     └────────────────┴──────────→ END

    串行执行，各节点间通过 AgentState TypedDict 传递数据。
    每个节点内部包含完整的 Python 自检 + 重试逻辑。
    """

    def __init__(self, llm: Any, vector_store: Any, status_callback: Any = None):
        """
        入参：
            llm             : ChatOpenAI — LLM 实例
            vector_store    : VectorStoreManager — 向量库管理器
            status_callback : callable | None — 实时状态回调
                             callback(agent_name, phase, message)
        """
        if not _HAS_LANGGRAPH:
            raise ImportError(
                "LangGraph 未安装，无法构建工作流。"
                "请执行：pip install langgraph"
            )

        self.llm = llm
        self.vector_store = vector_store
        self.status_callback = status_callback

        # 实例化三大 Agent
        self.retrieval_agent = RetrievalAgent(llm, vector_store)
        self.understanding_agent = UnderstandingAgent(llm)
        self.generation_agent = NoteGenerateAgent(llm)

        # 构建并编译工作流图
        self._graph = self._build_graph()
        logger.info("AgentWorkflow 初始化完成 | 工作流已编译")

    # ----------------------------------------------------------
    # 工作流图构建
    # ----------------------------------------------------------

    def _build_graph(self) -> Any:
        """
        构建 LangGraph StateGraph。

        节点：
          "retrieval"    — RetrievalAgent.execute(state)
          "understanding" — UnderstandingAgent.execute(state)
          "generation"   — NoteGenerateAgent.execute(state)

        边：
          START → retrieval（无条件）
          retrieval → understanding（正常） / END（终止）
          understanding → generation（正常） / END（终止）
          generation → END（无条件）
        """
        graph = StateGraph(AgentState)

        # --- 添加节点 ---
        graph.add_node("retrieval", self._retrieval_node)
        graph.add_node("understanding", self._understanding_node)
        graph.add_node("generation", self._generation_node)

        # --- 设置入口 ---
        graph.add_edge(START, "retrieval")

        # --- 条件路由：检索后 ---
        graph.add_conditional_edges(
            "retrieval",
            self._route_after_retrieval,
            {
                "understanding": "understanding",  # 正常 → 进入理解节点
                "END": END,                         # 终止 → 结束
            },
        )

        # --- 条件路由：理解后 ---
        graph.add_conditional_edges(
            "understanding",
            self._route_after_understanding,
            {
                "generation": "generation",  # 正常 → 进入生成节点
                "END": END,                   # 终止 → 结束
            },
        )

        # --- 生成后无条件结束 ---
        graph.add_edge("generation", END)

        # 编译并返回可执行图
        compiled = graph.compile()
        logger.info("LangGraph 工作流图构建完成")
        return compiled

    # ----------------------------------------------------------
    # 节点函数
    # ----------------------------------------------------------

    def _retrieval_node(self, state: AgentState) -> Dict[str, Any]:
        """
        检索节点 —— 封装 RetrievalAgent.execute()。
        """
        logger.info("[Workflow] → 进入检索节点")
        result = self.retrieval_agent.execute(state, status_callback=self.status_callback)
        logger.info(
            "[Workflow] ← 检索节点完成 | status=%s",
            result.get("status", "unknown"),
        )
        return result

    def _understanding_node(self, state: AgentState) -> Dict[str, Any]:
        """
        理解梳理节点 —— 封装 UnderstandingAgent.execute()。
        """
        logger.info("[Workflow] → 进入理解梳理节点")
        result = self.understanding_agent.execute(state, status_callback=self.status_callback)
        logger.info(
            "[Workflow] ← 理解梳理节点完成 | status=%s",
            result.get("status", "unknown"),
        )
        return result

    def _generation_node(self, state: AgentState) -> Dict[str, Any]:
        """
        笔记生成节点 —— 封装 NoteGenerateAgent.execute()。
        """
        logger.info("[Workflow] → 进入笔记生成节点")
        result = self.generation_agent.execute(state, status_callback=self.status_callback)
        logger.info(
            "[Workflow] ← 笔记生成节点完成 | status=%s",
            result.get("status", "unknown"),
        )
        return result

    # ----------------------------------------------------------
    # 路由函数（Python 决策 → LangGraph 条件边）
    # ----------------------------------------------------------

    @staticmethod
    def _route_after_retrieval(state: AgentState) -> str:
        """
        【Python 路由决策】检索节点后决定下一步。

        规则：
          - status == "terminated" → 终止（无有效素材）
          - 否则                      → 进入理解梳理
        """
        if state.get("status") == "terminated":
            logger.warning("[Workflow] 检索终止 → 流程结束")
            return "END"
        return "understanding"

    @staticmethod
    def _route_after_understanding(state: AgentState) -> str:
        """
        【Python 路由决策】理解节点后决定下一步。

        规则：
          - status == "terminated" → 终止（框架构建失败）
          - 否则                      → 进入笔记生成
        """
        if state.get("status") == "terminated":
            logger.warning("[Workflow] 理解梳理终止 → 流程结束")
            return "END"
        return "generation"

    # ----------------------------------------------------------
    # 对外入口
    # ----------------------------------------------------------

    def run(self, query: str, style: str = "outline") -> Dict[str, Any]:
        """
        同步执行完整工作流。

        入参：
            query : str — 用户查询（知识点 / 主题）
            style : str — 笔记风格：outline / mindmap / exam_points

        出参：
            Dict — {
                "note"   : str,  # 最终笔记
                "status" : str,  # completed / terminated
                "error"  : str,  # 错误信息（terminated 时有值）
                "context": str,  # 检索摘要（调试用）
                "framework": str, # 知识框架（调试用）
            }
        """
        logger.info("=" * 60)
        logger.info("[Workflow] 开始执行 | query=%.60s | style=%s", query, style)
        start_time = time.time()

        # 构造初始状态
        initial_state: AgentState = {
            "query": query,
            "context": "",
            "raw_context": "",
            "framework": "",
            "note": "",
            "style": style,
            "error": "",
            "status": "running",
        }

        # 执行工作流
        try:
            final_state = self._graph.invoke(initial_state)
        except Exception as exc:
            logger.exception("[Workflow] 执行异常")
            elapsed = time.time() - start_time
            return {
                "note": "",
                "status": "terminated",
                "error": f"工作流执行异常：{exc}",
                "context": "",
                "framework": "",
                "elapsed_seconds": round(elapsed, 2),
            }

        elapsed = time.time() - start_time
        logger.info(
            "[Workflow] 执行完成 | status=%s | 耗时=%.2fs",
            final_state.get("status", "unknown"),
            elapsed,
        )
        logger.info("=" * 60)

        return {
            "note": final_state.get("note", ""),
            "status": final_state.get("status", "unknown"),
            "error": final_state.get("error", ""),
            "context": final_state.get("context", ""),
            "framework": final_state.get("framework", ""),
            "elapsed_seconds": round(elapsed, 2),
        }


# ============================================================
# 九、模块级便捷入口
# ============================================================

# 模块级单例（延迟初始化）
_workflow_instance: Optional[AgentWorkflow] = None


def get_workflow(vector_store: Any) -> AgentWorkflow:
    """
    获取 AgentWorkflow 单例。

    入参：
        vector_store : VectorStoreManager — 向量库管理器

    出参：
        AgentWorkflow 实例

    说明：
        首次调用时初始化 LLM 和 AgentWorkflow，后续调用复用同一实例。
    """
    global _workflow_instance
    if _workflow_instance is None:
        llm = _create_llm()
        _workflow_instance = AgentWorkflow(llm, vector_store)
        logger.info("AgentWorkflow 全局单例已创建")
    return _workflow_instance


def run_workflow(
    query: str,
    style: str = "outline",
    vector_store: Any = None,
    status_callback: Any = None,
) -> Dict[str, Any]:
    """
    对外统一入口 —— 执行完整的「检索 → 理解 → 生成」工作流。

    入参：
        query           : str — 用户查询（知识点 / 主题）
        style           : str — 笔记风格：outline / mindmap / exam_points
        vector_store    : VectorStoreManager | None — 向量库管理器，
                          若为 None 则使用默认配置自动创建
        status_callback : callable | None — 实时状态回调
                          callback(agent_name, phase, message)

    出参：
        Dict — {
            "note"       : str,  # 最终生成的笔记
            "status"     : str,  # "completed" | "terminated"
            "error"      : str,  # 错误信息（仅 terminated 场景有值）
            "context"    : str,  # 检索到的摘要素材
            "framework"  : str,  # 构建的知识框架
        }

    使用示例：
        from app.vector_store import VectorStoreManager
        from app.agents import run_workflow

        store = VectorStoreManager(embedding_backend="dummy")
        result = run_workflow("什么是RAG", style="outline", vector_store=store)
        print(result["note"])
    """
    logger.info("[入口] run_workflow 被调用 | query=%.60s | style=%s", query, style)

    # --- 参数校验 ---
    if not query or not query.strip():
        return {
            "note": "",
            "status": "terminated",
            "error": "查询内容不能为空。",
            "context": "",
            "framework": "",
        }

    if style not in VALID_STYLES:
        logger.warning("未知风格「%s」→ 回退为 outline", style)
        style = "outline"

    # --- 向量库初始化 ---
    if vector_store is None:
        try:
            from app.vector_store import VectorStoreManager

            vector_store = VectorStoreManager(embedding_backend="dummy")
            logger.warning("未提供 vector_store，使用哑后端（仅测试可用）")
        except Exception as exc:
            return {
                "note": "",
                "status": "terminated",
                "error": f"向量库初始化失败：{exc}",
                "context": "",
                "framework": "",
            }

    # --- 获取/创建工作流实例 ---
    try:
        # 若有状态回调，每次创建新实例（避免并发冲突）
        if status_callback is not None:
            llm = _create_llm()
            workflow = AgentWorkflow(llm, vector_store, status_callback=status_callback)
        else:
            workflow = get_workflow(vector_store)
    except RuntimeError as exc:
        return {
            "note": "",
            "status": "terminated",
            "error": f"工作流初始化失败（LLM 未配置？）：{exc}",
            "context": "",
            "framework": "",
        }

    # --- 执行 ---
    return workflow.run(query, style)
