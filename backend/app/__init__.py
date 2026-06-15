"""
============================================================
app 包 — 基于职能型多Agent的本地知识库学习笔记生成助手
============================================================

核心架构（前后端分离）：
  - 后端：FastAPI + LangGraph + ChromaDB
  - 前端：原生 HTML / CSS / JS
  - 支持：TXT / PDF / MD 文件上传 → 向量库增量更新 → 三大职能 Agent 串行生成笔记

三大职能 Agent（串行流水线）：
  1. RetrievalAgent      — 检索 Agent：从向量库召回相关知识片段
  2. UnderstandingAgent  — 理解 Agent：分析知识结构，规划笔记大纲
  3. NoteGenerationAgent — 笔记生成 Agent：基于大纲生成结构化学习笔记

设计原则：
  - Agent 负责决策、自检、重试；LLM 仅作为文本生成工具
  - 向量库独立管理，支持增量更新不重索引
  - 所有配置统一在 config.py 中管理，敏感信息通过 .env 读取

模块说明：
  - config.py       : 全局配置常量
  - main.py         : FastAPI 应用入口 & 日志初始化
  - schemas.py      : Pydantic 请求 / 响应模型
  - vector_store.py : ChromaDB 向量库封装（入库 / 检索 / 文本提取）
  - agents.py       : 三大 Agent 定义 & 流水线编排（LangGraph）
"""

__version__ = "0.1.0"
__author__ = "AI Learning Notes Team"
