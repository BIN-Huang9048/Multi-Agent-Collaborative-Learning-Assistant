"""
============================================================
全局配置模块 (config.py)
============================================================
职责：
  1. 定义项目全局常量（文件存储路径、向量库路径、大小限制等）
  2. 管理 DeepSeek API 配置（密钥通过 .env 文件读取，此处仅保留占位符）
  3. 提供跨域（CORS）配置
  4. 启动时自动创建必要的本地目录

原则：
  - 所有路径使用相对路径（基于 backend/ 目录），兼容 Linux / macOS / Windows
  - 敏感信息一律从环境变量读取，硬编码仅写占位符
  - 模块级常量采用全大写命名，保持与 Python 社区惯例一致
============================================================
"""

import os
from pathlib import Path

# ----------------------------------------------------------
# 尝试加载 python-dotenv（若未安装则静默跳过，不阻断启动）
# ----------------------------------------------------------
try:
    from dotenv import load_dotenv

    # 按优先级查找 .env：backend/.env > 项目根/.env
    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
    else:
        load_dotenv()  # 回退到当前工作目录
except ImportError:
    pass  # dotenv 不是硬依赖；允许在不安装时继续使用系统环境变量

# ============================================================
# 一、项目路径定义
# ============================================================

# backend/ 目录（即本文件所在目录的父目录）
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# 项目根目录（backend/ 的上级目录）
PROJECT_DIR: Path = BASE_DIR.parent

# 用户上传的知识文件存储目录
KNOWLEDGE_DIR: Path = BASE_DIR / "knowledge"

# ChromaDB 向量库持久化存储目录
CHROMA_DIR: Path = BASE_DIR / "chroma_db"

# 应用日志文件存储目录
LOG_DIR: Path = BASE_DIR / "logs"

# ============================================================
# 二、文件上传限制
# ============================================================

# 单文件最大体积：20MB（单位：字节）
MAX_FILE_SIZE: int = 20 * 1024 * 1024

# 允许上传的文件扩展名集合（不含点号，统一小写比较）
ALLOWED_EXTENSIONS: set = {"txt", "pdf", "md"}

# ============================================================
# 三、DeepSeek API 配置
#     密钥等敏感信息从环境变量读取，此处仅提供占位符。
#     说明：DeepSeek API 完全兼容 OpenAI SDK 调用格式，
#           使用 langchain-openai 库（ChatOpenAI / OpenAIEmbeddings）对接。
# ============================================================

# API 密钥（务必在 .env 中设置真实值）
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key")

# API 基础 URL（DeepSeek 官方端点）
DEEPSEEK_API_BASE: str = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")

# DeepSeek 对话模型名称
DEEPSEEK_CHAT_MODEL: str = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")

# DeepSeek 向量嵌入模型名称
DEEPSEEK_EMBED_MODEL: str = os.getenv("DEEPSEEK_EMBED_MODEL", "deepseek-embed")

# 生成温度（0.0~2.0，越高越随机）
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# 单次生成最大 token 数（思维导图需要较大输出空间）
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "8192"))

# LLM 请求超时时间（秒）
LLM_REQUEST_TIMEOUT: int = int(os.getenv("LLM_REQUEST_TIMEOUT", "300"))

# ============================================================
# 四、跨域（CORS）配置
# ============================================================

# 允许的跨域来源列表（逗号分隔的字符串 → 列表）
# 开发环境可设为 "*" 允许所有来源
_CORS_RAW: str = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5500,null",
)
CORS_ALLOW_ORIGINS: list = [
    origin.strip() for origin in _CORS_RAW.split(",") if origin.strip()
]

# 是否允许携带凭证（Cookie / Authorization 头）
CORS_ALLOW_CREDENTIALS: bool = True

# 允许的 HTTP 方法
CORS_ALLOW_METHODS: list = ["*"]

# 允许的请求头
CORS_ALLOW_HEADERS: list = ["*"]

# ============================================================
# 五、FastAPI 服务配置
# ============================================================

# 服务监听地址（0.0.0.0 表示监听所有网卡）
SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")

# 服务监听端口
SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8000"))

# 调试模式开关（控制 FastAPI debug 参数 & 日志级别）
DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() in ("true", "1", "yes")

# ============================================================
# 六、向量库配置
# ============================================================

# ChromaDB 集合（Collection）名称
VECTOR_COLLECTION_NAME: str = os.getenv("VECTOR_COLLECTION_NAME", "knowledge")

# 默认检索返回条数（Top-K）
VECTOR_SEARCH_TOP_K: int = int(os.getenv("VECTOR_SEARCH_TOP_K", "5"))

# ============================================================
# 七、日志配置
# ============================================================

# 日志级别：DEBUG / INFO / WARNING / ERROR
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG" if DEBUG_MODE else "INFO")

# 日志格式（包含时间戳、模块名、日志级别、消息）
LOG_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"
)

# 日志日期格式
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# 日志文件最大大小（MB），超过后自动轮转
LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB

# 日志文件保留备份数量
LOG_BACKUP_COUNT: int = 5

# ============================================================
# 八、启动初始化：确保必要目录存在
# ============================================================
for _dir in (KNOWLEDGE_DIR, CHROMA_DIR, LOG_DIR):
    try:
        _dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # 极端情况（权限不足等）不阻断启动，由后续逻辑自行处理
        pass
