"""
============================================================
文件处理模块 (file_handler.py)
============================================================
职责：
  1. 对前端上传文件执行二次校验（格式、大小）
  2. 将合法文件持久化到 knowledge/ 目录
  3. 按文件类型（TXT / PDF / MD）提取纯文本内容
  4. 清洗提取后的文本（去乱码、去空行、去多余空白）
  5. 使用 LangChain 文本分割器将长文本切分为语义块

依赖：
  - PyPDF2          : PDF 文本提取
  - markdown        : Markdown → 纯文本转换
  - langchain.text_splitter : 文本分片

约定：
  - 所有方法在异常时主动捕获并抛出可读的错误信息
  - 分片默认参数：chunk_size=1000, chunk_overlap=200
  - 文件校验拒绝空文件（0 字节）
============================================================
"""

import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Optional

# ----------------------------------------------------------
# 第三方库导入（延迟加载 + 优雅降级）
#    部分依赖可能尚未安装（如 PyPDF2、langchain），在模块加载时不做强制检查，
#    仅在对应功能被实际调用时才抛出明确的 ImportError，方便按需安装。
# ----------------------------------------------------------

# ---- PyPDF2（PDF 文本提取）----
try:
    from PyPDF2 import PdfReader
    from PyPDF2.errors import PyPdfError

    _HAS_PYPDF2 = True
except ImportError:  # pragma: no cover
    PdfReader = None     # type: ignore
    PyPdfError = None    # type: ignore
    _HAS_PYPDF2 = False

# ---- markdown（Markdown → 纯文本）----
try:
    import markdown as md_lib

    _HAS_MARKDOWN = True
except ImportError:  # pragma: no cover
    md_lib = None        # type: ignore
    _HAS_MARKDOWN = False

# ---- langchain.text_splitter（文本分片，兼容 langchain 0.x / 1.x）----
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    _HAS_LANGCHAIN = True
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        _HAS_LANGCHAIN = True
    except ImportError:  # pragma: no cover
        RecursiveCharacterTextSplitter = None  # type: ignore
        _HAS_LANGCHAIN = False

# ----------------------------------------------------------
# 项目内部模块
# ----------------------------------------------------------
from app.config import (
    ALLOWED_EXTENSIONS,  # 允许的扩展名集合，不含点号：{"txt", "pdf", "md"}
    MAX_FILE_SIZE,       # 单文件最大字节数（默认 20MB）
    KNOWLEDGE_DIR,       # 文件落盘目录
)

# ----------------------------------------------------------
# 模块级日志
# ----------------------------------------------------------
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# 模块级常量
# ----------------------------------------------------------
# 文本清洗正则（编译一次，反复使用）
_RE_CONTROL_CHARS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)  # 控制字符（保留 \n \t）
_RE_MULTI_SPACES = re.compile(r"[ \t]+")          # 连续空格/Tab
_RE_MULTI_NEWLINES = re.compile(r"\n{3,}")        # 连续 3+ 换行 → 压缩为 2 个
_RE_HTML_TAG = re.compile(r"<[^>]+>")             # HTML 标签（MD→HTML 后剥离用）
_RE_UNICODE_REPLACEMENT = re.compile(r"�")    # Unicode 替换字符（乱码标志）
_RE_PRIVATE_USE = re.compile(r"[-]")   # 私有区字符（非标准内容）
_RE_ZERO_WIDTH = re.compile(r"[​-‏ - ⁠-⁯﻿]")  # 零宽/不可见字符


class FileHandler:
    """
    文件处理器 —— 封装文件校验、存储、文本提取、清洗、分片的完整流水线。

    使用示例：
        handler = FileHandler()
        handler.check_file(upload_file, "笔记.pdf")
        saved_path = handler.save_file(upload_file, KNOWLEDGE_DIR)
        raw_text = handler.extract_text(saved_path)
        clean_text = handler.clean_text(raw_text)
        chunks = handler.split_text(clean_text)
    """

    # ============================================================
    # 构造方法
    # ============================================================

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        初始化文件处理器。

        入参：
            chunk_size   : int — 文本分片的目标大小（字符数），默认 1000
            chunk_overlap: int — 相邻分片的重叠字符数，默认 200
        """
        # 从 config 读取文件限制参数
        self.allowed_extensions: set = ALLOWED_EXTENSIONS
        self.max_file_size: int = MAX_FILE_SIZE
        self.knowledge_dir: Path = KNOWLEDGE_DIR

        # 文本分片参数
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap

        # 延迟初始化文本分割器（首次调用 split_text 时创建）
        self._splitter: Optional[RecursiveCharacterTextSplitter] = None

        # 确保知识目录存在
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "FileHandler 初始化完成 | 允许格式=%s | 最大体积=%dMB | 分片大小=%d | 重叠=%d",
            self.allowed_extensions,
            self.max_file_size // (1024 * 1024),
            self.chunk_size,
            self.chunk_overlap,
        )

    # ============================================================
    # 属性：延迟创建 LangChain 分割器
    # ============================================================

    @property
    def splitter(self) -> RecursiveCharacterTextSplitter:
        """
        获取文本分割器实例（延迟初始化）。

        出参：
            RecursiveCharacterTextSplitter — 配置好的 LangChain 分割器

        抛出：
            ImportError — langchain 未安装

        说明：
            使用 RecursiveCharacterTextSplitter 作为默认分割器，
            其按「段落 → 句子 → 字符」的优先级递归切分，
            在保持语义完整性的前提下尽量接近 chunk_size。
        """
        if not _HAS_LANGCHAIN:
            raise ImportError(
                "langchain 库未安装，无法使用文本分片功能。"
                "请执行：pip install langchain"
            )
        if self._splitter is None:
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=[
                    "\n\n",    # 段落边界（最高优先级）
                    "\n",      # 换行
                    "。",      # 中文句号
                    ". ",      # 英文句号+空格
                    "；",      # 中文分号
                    "; ",      # 英文分号+空格
                    "，",      # 中文逗号
                    ", ",      # 英文逗号+空格
                    " ",       # 空格（最低优先级）
                    "",        # 逐字符切分（兜底）
                ],
                length_function=len,
                is_separator_regex=False,
            )
            logger.debug(
                "文本分割器已创建 | chunk_size=%d | chunk_overlap=%d",
                self.chunk_size,
                self.chunk_overlap,
            )
        return self._splitter

    # ============================================================
    # 方法 1：文件二次校验
    # ============================================================

    def check_file(self, file, filename: str) -> None:
        """
        【方法1】前端上传文件的二次校验 —— 检查格式与大小。

        校验规则：
          a) 文件扩展名必须在 ALLOWED_EXTENSIONS 集合中
          b) 文件大小不得超过 MAX_FILE_SIZE（20MB）
          c) 文件不得为空（0 字节）

        入参：
            file     : fastapi.UploadFile — 上传文件对象（需保留 .file 可读）
            filename : str               — 原始文件名（含扩展名）

        抛出：
            ValueError  — 格式不支持 / 文件过大 / 文件为空
            IOError     — 文件不可读

        说明：
            校验完毕后会将文件指针复位到开头，确保后续 save_file 可正常读取。
        """
        # --------------------------------------------------------
        # 校验 1：文件扩展名
        # --------------------------------------------------------
        if not filename or "." not in filename:
            raise ValueError(
                f"文件名无效：「{filename or '(空)'}」，无法识别文件类型。"
            )

        # 提取扩展名，去掉前导点号，统一转为小写
        extension = Path(filename).suffix.lstrip(".").lower()

        if extension not in self.allowed_extensions:
            allowed_display = "、".join(
                sorted(self.allowed_extensions)
            )
            raise ValueError(
                f"不支持的文件格式「.{extension}」，"
                f"仅支持以下格式：{allowed_display}"
            )

        logger.debug("文件格式校验通过 | 文件名=%s | 扩展名=%s", filename, extension)

        # --------------------------------------------------------
        # 校验 2：文件大小（读取文件内容并检查长度）
        # --------------------------------------------------------
        try:
            # FastAPI UploadFile 的底层文件对象
            file_content: bytes = file.file.read()
        except Exception as exc:
            raise IOError(f"无法读取文件内容：{exc}") from exc

        file_size: int = len(file_content)

        # 校验 2a：空文件
        if file_size == 0:
            raise ValueError("文件内容为空，请上传有效文件。")

        # 校验 2b：超出大小限制
        if file_size > self.max_file_size:
            max_mb = self.max_file_size // (1024 * 1024)
            actual_mb = file_size / (1024 * 1024)
            raise ValueError(
                f"文件体积超出限制：{actual_mb:.1f}MB > {max_mb}MB（上限）"
            )

        # 复位文件指针，供后续 save_file 读取
        file.file.seek(0)

        logger.info(
            "文件校验通过 | 文件名=%s | 大小=%.2fKB | 格式=%s",
            filename,
            file_size / 1024,
            extension,
        )

    # ============================================================
    # 方法 2：保存文件
    # ============================================================

    def save_file(self, file, save_dir: Optional[Path] = None) -> Path:
        """
        【方法2】将上传文件持久化到本地目录。

        入参：
            file     : fastapi.UploadFile — 上传文件对象
            save_dir : Path | None        — 目标目录，默认使用 config.KNOWLEDGE_DIR

        出参：
            Path — 文件落盘后的完整路径

        抛出：
            IOError       — 目录不可写 / 磁盘空间不足
            ValueError    — 文件名非法

        说明：
            - 自动对文件名做安全清洗（移除路径穿越字符、统一编码）
            - 同名文件直接覆盖
        """
        target_dir = save_dir if save_dir is not None else self.knowledge_dir

        # --- 确保目标目录存在 ---
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise IOError(f"无法创建存储目录「{target_dir}」：{exc}") from exc

        # --- 安全清洗文件名 ---
        safe_name = self._sanitize_filename(file.filename)
        if not safe_name:
            raise ValueError(f"文件名清洗后为空：「{file.filename}」")

        destination = target_dir / safe_name

        # --- 写入文件 ---
        try:
            content = file.file.read()
            destination.write_bytes(content)
        except OSError as exc:
            raise IOError(f"文件写入失败「{destination}」：{exc}") from exc
        finally:
            # 确保文件指针复位
            try:
                file.file.seek(0)
            except Exception:
                pass

        logger.info(
            "文件保存成功 | 文件名=%s | 路径=%s | 大小=%d 字节",
            safe_name,
            destination,
            len(content),
        )
        return destination

    # ============================================================
    # 方法 3：提取纯文本
    # ============================================================

    def extract_text(self, file_path: Path) -> str:
        """
        【方法3】根据文件扩展名解析 TXT / PDF / MD，提取纯文本内容。

        入参：
            file_path : Path — 文件路径（需存在且可读）

        出参：
            str — 提取的纯文本内容（UTF-8 编码）

        抛出：
            FileNotFoundError — 文件不存在
            ValueError        — 不支持的文件格式
            IOError           — 文件读取失败 / PDF 解析失败

        支持格式：
            .txt  — 直接以 UTF-8 读取
            .md   — 先读取原始 Markdown，再转为纯文本（剥离格式标记）
            .pdf  — 使用 PyPDF2 逐页提取
        """
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在：{file_path}")

        if not file_path.is_file():
            raise ValueError(f"路径不是文件：{file_path}")

        # --- 获取扩展名（无点号、小写） ---
        extension = file_path.suffix.lstrip(".").lower()

        if extension not in self.allowed_extensions:
            raise ValueError(
                f"不支持的文件格式「.{extension}」，"
                f"仅支持：{', '.join(sorted(self.allowed_extensions))}"
            )

        # --- 按类型分发 ---
        logger.debug("开始文本提取 | 文件=%s | 格式=%s", file_path.name, extension)

        try:
            if extension == "txt":
                text = self._extract_txt(file_path)
            elif extension == "md":
                text = self._extract_md(file_path)
            elif extension == "pdf":
                text = self._extract_pdf(file_path)
            else:
                # 理论上不应到达此处（已在上面校验）
                raise ValueError(f"未处理的文件格式：{extension}")
        except Exception as exc:
            logger.error("文本提取失败 | 文件=%s | 错误=%s", file_path.name, exc)
            raise

        logger.info(
            "文本提取完成 | 文件=%s | 字符数=%d",
            file_path.name,
            len(text),
        )
        return text

    # ============================================================
    # 方法 4：文本清洗
    # ============================================================

    def clean_text(self, text: str) -> str:
        """
        【方法4】清洗提取后的文本 —— 去除乱码、空行、多余空白。

        清洗步骤：
          1. Unicode 规范化（NFKC）
          2. 去除零宽字符 & 不可见控制字符
          3. 去除 Unicode 私有区字符（非标准内容）
          4. 将 Unicode 替换字符（�）替换为空
          5. 统一换行符为 \n
          6. 压缩连续空格 / Tab 为单个空格
          7. 按行去除首尾空白，删除纯空白行
          8. 压缩连续 3+ 空行为双空行（保留段落间距）

        入参：
            text : str — 原始文本

        出参：
            str — 清洗后的干净文本
        """
        if not text or not text.strip():
            logger.debug("文本清洗跳过：输入为空或仅含空白字符")
            return ""

        original_len = len(text)

        # ---- 步骤 1：Unicode 规范化 ----
        text = unicodedata.normalize("NFKC", text)

        # ---- 步骤 2：去除零宽/不可见字符 ----
        text = _RE_ZERO_WIDTH.sub("", text)

        # ---- 步骤 3：去除控制字符（保留 \n 和 \t）----
        text = _RE_CONTROL_CHARS.sub("", text)

        # ---- 步骤 4：去除 Unicode 私有区字符 ----
        text = _RE_PRIVATE_USE.sub("", text)

        # ---- 步骤 5：替换 Unicode 替换字符（乱码标记）----
        text = _RE_UNICODE_REPLACEMENT.sub("", text)

        # ---- 步骤 6：统一换行符 ----
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # ---- 步骤 7：压缩连续空格/Tab ----
        text = _RE_MULTI_SPACES.sub(" ", text)

        # ---- 步骤 8：逐行清洗 ----
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped:  # 跳过纯空白行
                cleaned_lines.append(stripped)

        text = "\n".join(cleaned_lines)

        # ---- 步骤 9：压缩连续 3+ 空行为双空行 ----
        text = _RE_MULTI_NEWLINES.sub("\n\n", text)

        # ---- 步骤 10：首尾去空白 ----
        text = text.strip()

        removed_chars = original_len - len(text)
        logger.debug(
            "文本清洗完成 | 原始=%d字符 | 清洗后=%d字符 | 去除=%d字符 (%.1f%%)",
            original_len,
            len(text),
            removed_chars,
            (removed_chars / original_len * 100) if original_len else 0,
        )
        return text

    # ============================================================
    # 方法 5：文本分片
    # ============================================================

    def split_text(self, text: str) -> List[str]:
        """
        【方法5】使用 LangChain RecursiveCharacterTextSplitter 将长文本切分为块。

        入参：
            text : str — 待分片的文本（通常已经过 clean_text 清洗）

        出参：
            List[str] — 文本块列表，每个元素为一个分片字符串

        说明：
            - 分片参数在 __init__ 中设定（默认 chunk_size=1000, chunk_overlap=200）
            - 相邻分片保持 200 字符重叠以维持语义连续性
            - 若输入文本短于 chunk_size，返回包含全文的单元素列表
            - 空文本返回空列表
        """
        if not text or not text.strip():
            logger.debug("文本分片跳过：输入为空")
            return []

        original_len = len(text)

        try:
            chunks: List[str] = self.splitter.split_text(text)
        except Exception as exc:
            logger.error("文本分片失败 | 错误=%s", exc)
            raise RuntimeError(f"文本分片过程出错：{exc}") from exc

        # 过滤可能的空分片
        chunks = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

        logger.info(
            "文本分片完成 | 原始=%d字符 | 分片数=%d | 平均=%d字符/片",
            original_len,
            len(chunks),
            sum(len(c) for c in chunks) // max(len(chunks), 1),
        )
        return chunks

    # ============================================================
    # 便捷方法：完整流水线
    # ============================================================

    def process(
        self,
        file,
        filename: str,
        save_dir: Optional[Path] = None,
        skip_clean: bool = False,
        skip_split: bool = False,
    ) -> dict:
        """
        【便捷方法】一键执行「校验 → 保存 → 提取 → 清洗 → 分片」全流程。

        入参：
            file       : UploadFile  — 前端上传的文件对象
            filename   : str        — 原始文件名
            save_dir   : Path|None  — 保存目录（默认 KNOWLEDGE_DIR）
            skip_clean : bool       — 跳过文本清洗步骤
            skip_split : bool       — 跳过文本分片步骤

        出参：
            dict — {
                "saved_path" : Path,          # 文件保存路径
                "raw_text"   : str,           # 原始提取文本
                "clean_text" : str,           # 清洗后文本（skip_clean=True 时与 raw_text 相同）
                "chunks"     : List[str],     # 文本分片列表（skip_split=True 时为空列表）
            }
        """
        logger.info("==== 文件处理流水线开始 | 文件=%s ====", filename)

        # Step 1: 校验
        self.check_file(file, filename)

        # Step 2: 保存
        saved_path = self.save_file(file, save_dir)

        # Step 3: 提取文本
        raw_text = self.extract_text(saved_path)

        # Step 4: 清洗（可选跳过）
        if skip_clean:
            clean_text = raw_text
            logger.debug("跳过文本清洗步骤（skip_clean=True）")
        else:
            clean_text = self.clean_text(raw_text)

        # Step 5: 分片（可选跳过）
        if skip_split:
            chunks = []
            logger.debug("跳过文本分片步骤（skip_split=True）")
        else:
            chunks = self.split_text(clean_text)

        logger.info(
            "==== 文件处理流水线完成 | 文件=%s | 原始=%d字符 | 清洗=%d字符 | 分片=%d块 ====",
            filename,
            len(raw_text),
            len(clean_text),
            len(chunks),
        )

        return {
            "saved_path": saved_path,
            "raw_text": raw_text,
            "clean_text": clean_text,
            "chunks": chunks,
        }

    # ============================================================
    # 私有方法：各格式文本提取器
    # ============================================================

    @staticmethod
    def delete_physical_file(filename: str, base_dir: Optional[Path] = None) -> bool:
        """
        删除知识目录中的物理文件。

        入参：
            filename : str        — 要删除的文件名（需与保存时的名称一致）
            base_dir : Path|None  — 知识目录，默认使用 config.KNOWLEDGE_DIR

        出参：
            bool — 删除成功返回 True，文件不存在返回 False

        抛出：
            OSError — 文件存在但删除失败（权限不足等）
        """
        from app.config import KNOWLEDGE_DIR as _default_dir

        target_dir = base_dir if base_dir is not None else _default_dir
        file_path = target_dir / filename

        if not file_path.exists():
            logger.debug("物理文件不存在，跳过删除 | 路径=%s", file_path)
            return False

        try:
            file_path.unlink()
            logger.info("物理文件已删除 | 文件=%s", filename)
            return True
        except OSError as exc:
            logger.exception("物理文件删除失败 | 文件=%s", filename)
            raise OSError(f"文件删除失败「{filename}」：{exc}") from exc

    @staticmethod
    def _extract_txt(file_path: Path) -> str:
        """
        提取 TXT 纯文本。

        入参：file_path : Path — TXT 文件路径
        出参：str — 文本内容
        抛出：IOError — 读取失败
        """
        try:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except UnicodeDecodeError:
            # UTF-8 失败时尝试 GBK（兼容中文 Windows 导出的文本）
            try:
                return file_path.read_text(encoding="gbk", errors="replace")
            except Exception as exc:
                raise IOError(f"TXT 文件编码无法识别：{exc}") from exc
        except OSError as exc:
            raise IOError(f"TXT 文件读取失败：{exc}") from exc

    @staticmethod
    def _extract_md(file_path: Path) -> str:
        """
        提取 Markdown 纯文本 —— 剥离格式标记。

        入参：file_path : Path — MD 文件路径
        出参：str — 去格式后的纯文本
        抛出：IOError — 读取失败

        处理策略：
          1. 读取原始 Markdown 文本
          2. 若 markdown 库可用，将其转为 HTML 再剥离标签
          3. 若 markdown 库不可用，回退为原始文本（保留格式标记）
        """
        try:
            raw_md = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise IOError(f"Markdown 文件读取失败：{exc}") from exc

        if _HAS_MARKDOWN:
            try:
                # Markdown → HTML
                html = md_lib.markdown(
                    raw_md,
                    extensions=["extra", "sane_lists"],
                )
                # HTML → 纯文本（去除所有标签）
                plain_text = _RE_HTML_TAG.sub("", html)
                # HTML 实体解码（常见的）
                plain_text = (
                    plain_text.replace("&amp;", "&")
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                    .replace("&quot;", '"')
                    .replace("&#39;", "'")
                    .replace("&nbsp;", " ")
                )
                return plain_text
            except Exception:
                logger.warning(
                    "Markdown 转换失败，回退为原始文本 | 文件=%s",
                    file_path.name,
                )
                return raw_md
        else:
            logger.debug(
                "markdown 库未安装，返回原始 Markdown 文本 | 文件=%s",
                file_path.name,
            )
            return raw_md

    @staticmethod
    def _extract_pdf(file_path: Path) -> str:
        """
        提取 PDF 纯文本 —— 使用 PyPDF2 逐页提取。

        入参：file_path : Path — PDF 文件路径
        出参：str — 所有页面的文本拼接（页间用双换行分隔）
        抛出：ImportError — PyPDF2 未安装
              IOError    — 文件不可读 / PDF 损坏 / 加密

        说明：
          - 自动跳过无法提取文本的页面（如图片页）
          - 加密 PDF 会抛出明确错误
        """
        if not _HAS_PYPDF2:
            raise ImportError(
                "PyPDF2 库未安装，无法提取 PDF 文本。"
                "请执行：pip install PyPDF2"
            )
        try:
            reader = PdfReader(str(file_path))
        except PyPdfError as exc:
            raise IOError(f"PDF 文件解析失败（可能已损坏）：{exc}") from exc
        except OSError as exc:
            raise IOError(f"PDF 文件读取失败：{exc}") from exc

        # 检查是否加密
        if reader.is_encrypted:
            raise IOError(
                f"PDF 文件已加密，无法提取文本：「{file_path.name}」"
            )

        total_pages = len(reader.pages)
        if total_pages == 0:
            logger.warning("PDF 文件无页面 | 文件=%s", file_path.name)
            return ""

        page_texts: List[str] = []
        empty_pages = 0

        for i, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    page_texts.append(page_text.strip())
                else:
                    empty_pages += 1
                    logger.debug(
                        "PDF 第 %d/%d 页无文本内容", i, total_pages
                    )
            except Exception as exc:
                logger.warning(
                    "PDF 第 %d/%d 页提取失败：%s", i, total_pages, exc
                )
                empty_pages += 1

        if empty_pages == total_pages:
            logger.warning(
                "PDF 所有页面均无文本 | 文件=%s | 页数=%d",
                file_path.name,
                total_pages,
            )

        result = "\n\n".join(page_texts)

        logger.debug(
            "PDF 提取完成 | 文件=%s | 总页=%d | 有文本页=%d | 总字符=%d",
            file_path.name,
            total_pages,
            len(page_texts),
            len(result),
        )
        return result

    # ============================================================
    # 私有方法：文件名安全清洗
    # ============================================================

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        清洗文件名 —— 移除路径穿越字符、非法字符、前后空白。

        入参：
            filename : str — 原始文件名（可能含路径信息）

        出参：
            str — 安全的文件名

        说明：
            - 去除路径分隔符（防止目录穿越攻击）
            - 非法字符替换为下划线
            - 保留中文、字母、数字、点号、下划线、连字符
        """
        if not filename:
            return ""

        # 去除路径信息，仅保留文件名部分
        filename = os.path.basename(filename)

        # 去除前后空白
        filename = filename.strip()

        # 替换非法字符为下划线
        # 保留：ASCII 字母数字 (\w)、中文汉字区间 (一-鿿)、点号、连字符
        filename = re.sub(
            r"[^\w\.\-一-鿿]",
            "_",
            filename,
        )

        # 去除连续下划线
        filename = re.sub(r"_{2,}", "_", filename)

        # 去除首尾下划线和点号（但保留扩展名前的点号）
        filename = filename.strip("_")

        # 确保文件名非空
        if not filename or filename in (".", ".."):
            filename = "unnamed_file"

        return filename
