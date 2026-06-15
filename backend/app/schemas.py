"""
============================================================
Pydantic 数据模型 (schemas.py)
============================================================
职责：
  1. 定义请求/响应的数据结构
  2. 提供字段级别的自动校验（非空、长度、枚举）
  3. 统一前后端 API 契约

模型列表：
  - GenerateNoteRequest  : 笔记生成请求
  - GenerateNoteResponse : 笔记生成响应
  - UploadFileResponse   : 文件上传响应
  - ErrorResponse        : 统一错误响应
============================================================
"""

from typing import Optional
from pydantic import BaseModel, Field


# ============================================================
# 笔记生成请求
# ============================================================

class GenerateNoteRequest(BaseModel):
    """
    笔记生成请求体。

    字段：
      query : str — 用户查询的知识点/主题（必填，1~500 字符）
      style : str — 笔记风格，可选 outline / mindmap / exam_points
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="用户查询的知识点/主题",
        examples=["请解释什么是RAG（检索增强生成）"],
    )
    style: str = Field(
        "outline",
        pattern=r"^(outline|mindmap|exam_points)$",
        description="笔记风格：outline(提纲) / mindmap(思维导图) / exam_points(考点清单)",
        examples=["outline"],
    )


# ============================================================
# 笔记生成响应
# ============================================================

class GenerateNoteResponse(BaseModel):
    """
    笔记生成成功响应体。

    字段：
      code      : int  — 状态码（200=成功）
      msg       : str  — 状态描述
      context   : str  — 检索到的原文素材摘要
      framework : str  — 知识框架
      note      : str  — 最终生成的笔记
      style     : str  — 使用的笔记风格
      elapsed   : float — 工作流耗时（秒）
    """
    code: int = 200
    msg: str = "笔记生成成功"
    context: str = ""
    framework: str = ""
    note: str = ""
    style: str = "outline"
    elapsed: float = 0.0


# ============================================================
# 文件上传响应
# ============================================================

class UploadFileResponse(BaseModel):
    """
    文件上传响应体。

    字段：
      code         : int  — 状态码（200=成功）
      msg          : str  — 状态描述
      filename     : str  — 保存后的文件名
      chunks_count : int  — 入库的文本分片数
    """
    code: int = 200
    msg: str = "文件上传并入库成功"
    filename: str = ""
    chunks_count: int = 0


# ============================================================
# 笔记下载请求
# ============================================================

class DownloadNoteRequest(BaseModel):
    """
    笔记下载请求体。

    字段：
      content    : str — 笔记内容（Markdown 文本）
      style      : str — 笔记风格：outline / mindmap / exam_points
      svg_markup : str — 思维导图 SVG 标签（仅 mindmap 需要，前端序列化后传入）
      title      : str — 笔记标题（用于文件名和文档标题）
    """
    content: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="笔记内容",
    )
    style: str = Field(
        ...,
        pattern=r"^(outline|mindmap|exam_points)$",
        description="笔记风格",
    )
    svg_markup: str = Field(
        "",
        max_length=200000,
        description="思维导图 SVG 标记（仅 mindmap 风格需要）",
    )
    title: str = Field(
        "学习笔记",
        max_length=200,
        description="笔记标题",
    )


# ============================================================
# 统一错误响应
# ============================================================

class ErrorResponse(BaseModel):
    """
    统一错误响应体 —— 所有异常情况均返回此格式。

    字段：
      code       : int  — HTTP 状态码（4xx/5xx）
      msg        : str  — 人类可读的错误描述
      detail     : str  — 详细错误信息（可选，调试用）
      error_type : str  — 异常类型名（可选）
    """
    code: int
    msg: str
    detail: Optional[str] = None
    error_type: Optional[str] = None
