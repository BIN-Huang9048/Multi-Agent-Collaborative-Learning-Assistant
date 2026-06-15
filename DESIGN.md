# 🏗️ 设计文档 — 本地知识库学习笔记生成助手

> **版本**: 0.1.0  
> **最后更新**: 2026-06-15  
> **文档范围**: 系统架构、数据流、模块设计、关键决策、接口协议、安全考量

---

## 目录

1. [概述与目标](#1-概述与目标)
2. [系统架构](#2-系统架构)
3. [数据流设计](#3-数据流设计)
4. [前端设计](#4-前端设计)
5. [后端模块设计](#5-后端模块设计)
6. [多 Agent 工作流设计](#6-多-agent-工作流设计)
7. [向量库设计](#7-向量库设计)
8. [文件处理管线](#8-文件处理管线)
9. [笔记下载设计](#9-笔记下载设计)
10. [API 协议规范](#10-api-协议规范)
11. [状态管理](#12-状态管理)
12. [错误处理策略](#13-错误处理策略)
13. [安全考量](#14-安全考量)
14. [环境与部署](#15-环境与部署)
15. [关键技术决策](#16-关键技术决策)

---

## 1. 概述与目标

### 1.1 项目定位

面向个人学习场景的本地知识库工具。用户上传自己的学习资料，系统自动建立向量索引，然后根据用户查询主题，通过三个职能化 Agent 串行协作生成结构化学习笔记。

### 1.2 核心设计目标

| 目标 | 实现方式 |
|------|----------|
| **前后端分离** | 前端纯静态 HTML/CSS/JS，后端 FastAPI REST + SSE |
| **私有知识库** | 本地 ChromaDB 持久化向量存储，数据不外传 |
| **Agent 协作** | 检索 → 理解 → 生成三阶段串行流水线 |
| **LLM 安全** | LLM 仅作文本工具，所有决策/校验/重试由 Python 完成 |
| **实时反馈** | SSE 流式推送每个 Agent 的工作状态 |
| **防幻觉** | 溯源校验 + 格式合规 + 内容降噪三道防线 |
| **离线 Embedding** | 使用本地模型 `BAAI/bge-small-zh-v1.5` 做嵌入，不依赖外部 API |

### 1.3 非目标

- **不**支持多用户 / 权限管理（单用户本地工具）
- **不**支持实时协作编辑
- **不**对接外部知识库或搜索引擎
- **不**做 GPU 加速（CPU 推理即可满足个人量级）

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser (Frontend)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ 文件上传  │  │ 笔记生成  │  │ 结果展示  │  │ 思维导图 SVG 渲染 │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────────┬─────────┘ │
└───────┼──────────────┼──────────────┼─────────────────┼──────────┘
        │ FormData     │ JSON/SSE     │ JSON            │ SVG→JSON
        ▼              ▼              ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Server (:8000)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     Middleware Layer                       │   │
│  │  CORS → RequestLogging → GlobalExceptionHandler            │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ /api/upload  │  │/api/generate │  │ /api/download_note   │  │
│  │    _file     │  │    _note     │  │                      │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │               │
│         ▼                 ▼                      ▼               │
│  ┌──────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│  │FileHandler│  │ AgentWorkflow    │  │ DOCX / PDF Gen   │      │
│  │          │  │ (LangGraph)      │  │                  │      │
│  │• 校验    │  │                  │  │ • python-docx     │      │
│  │• 提取    │  │ RetrievalAgent   │  │ • svglib+reportlab│      │
│  │• 清洗    │  │       ↓          │  └──────────────────┘      │
│  │• 分片    │  │ UnderstandingAg  │                             │
│  └────┬─────┘  │       ↓          │                             │
│       │        │ NoteGenerateAg   │                             │
│       ▼        └────────┬─────────┘                             │
│  ┌──────────┐           │                                       │
│  │VectorStore│←──────────┘                                      │
│  │ (ChromaDB)│                                                  │
│  └──────────┘                                                   │
└─────────────────────────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
┌──────────────┐  ┌─────────────────────┐
│ ChromaDB     │  │ DeepSeek API        │
│ (SQLite+向量)│  │ (OpenAI 兼容协议)    │
└──────────────┘  └─────────────────────┘
```

### 2.2 技术分层

```
┌─────────────────────────────────────────┐
│ 表现层  │ HTML5 + CSS3 + Vanilla JS     │
├─────────────────────────────────────────┤
│ 接口层  │ FastAPI + Pydantic Schema     │
├─────────────────────────────────────────┤
│ 业务层  │ AgentWorkflow + FileHandler   │
├─────────────────────────────────────────┤
│ 编排层  │ LangGraph StateGraph          │
├─────────────────────────────────────────┤
│ 模型层  │ DeepSeek ChatOpenAI           │
├─────────────────────────────────────────┤
│ 数据层  │ ChromaDB + JSON + 文件系统    │
└─────────────────────────────────────────┘
```

### 2.3 进程模型

单进程模型，关键单例：

- `FileHandler()` — 文件处理
- `VectorStoreManager()` — 向量库（线程安全，内含 `threading.Lock`）
- `AgentWorkflow()` — Agent 编排（按需创建 LLM 实例）

笔记生成在**后台线程**中执行，通过 `asyncio.Queue` 与主事件循环通信，实现 SSE 非阻塞推送。

---

## 3. 数据流设计

### 3.1 文档入库流

```
┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 浏览器    │───▶│ POST         │───▶│ check_file│───▶│ save_file│───▶│extract   │
│ 拖拽上传  │    │ /upload_file │    │ (校验)    │    │ (落盘)    │    │_text()   │
└──────────┘    └─────────────┘    └──────────┘    └──────────┘    └────┬─────┘
                                                                        │
                                    ┌──────────┐    ┌──────────┐        │
      ChromaDB ◀────────────────────│add_      │◀───│split_text│◀───────┘
      (持久化)                       │documents │    │(1000/200)│   clean_text()
                                    └──────────┘    └──────────┘
```

各阶段详情：

| 阶段 | 模块 | 方法 | 输入 | 输出 | 关键逻辑 |
|------|------|------|------|------|----------|
| 1. 校验 | FileHandler | `check_file()` | UploadFile | None / raise | 扩展名∈{txt,md,pdf}，size≤20MB，非空 |
| 2. 落盘 | FileHandler | `save_file()` | UploadFile, dir | Path | 文件名净化（去除路径分隔符等危险字符） |
| 3. 提取 | FileHandler | `extract_text()` | Path | str | 按扩展名分发：TXT(UTF-8/GBK fallback)、MD(markdown库)、PDF(PyPDF2逐页) |
| 4. 清洗 | FileHandler | `clean_text()` | str | str | Unicode标准化→控制字符移除→零宽字符移除→空白压缩→换行合并 |
| 5. 分片 | FileHandler | `split_text()` | str | List[str] | LangChain RecursiveCharacterTextSplitter，chunk_size=1000, overlap=200 |
| 6. 入库 | VectorStoreManager | `add_documents()` | List[str], source_name | int(chunks_count) | 逐片向量化→UUID ID→批量 upsert→返回入库数 |

### 3.2 笔记生成流（核心）

```
POST /api/generate_note
  │
  ├── 参数校验（Pydantic + 手动）→ 400 拦截
  │
  ├── 获取 VectorStoreManager 单例
  │
  ├── 创建 asyncio.Queue（SSE 事件中转）
  │
  ├── 启动后台线程 ──────────────────────────────────────┐
  │                                                     │
  │   run_workflow(query, style, vector_store, cb)       │
  │     │                                                │
  │     ├── RetrievalAgent.execute(state, cb)            │
  │     │   ├── [Python] 输入校验                        │
  │     │   ├── [Python] 向量检索 (top_k=5)              │
  │     │   ├── [Python] 相关性自检 (distance≤0.8 +      │
  │     │   │   关键词重合度≥30%)                         │
  │     │   ├── [Python] 不足→二次检索(distance≤1.0)     │
  │     │   ├── [Python] 仍不足→终止(Sorry提示)          │
  │     │   └── [LLM] 摘要提纯(≤800字)                   │
  │     │                                                │
  │     ├── UnderstandingAgent.execute(state, cb)        │
  │     │   ├── [Python] 空素材拦截→终止                 │
  │     │   ├── [LLM] 构建知识框架(层级结构)             │
  │     │   ├── [Python] 溯源校验(5级匹配)               │
  │     │   ├── [Python] 框架合规(≥2节,深度≥2)          │
  │     │   └── [Python] 失败→重试(最多1次)             │
  │     │                                                │
  │     └── NoteGenerateAgent.execute(state, cb)         │
  │         ├── [Python] 空框架拦截→终止                 │
  │         ├── [LLM] 套用模板生成笔记                   │
  │         ├── [Python] 格式正则校验                    │
  │         ├── [Python] 内容降噪(AI自述/重复率)        │
  │         └── [Python] 失败→重试(最多2次)→降级        │
  │                                                     │
  │   各 Agent 每步调 cb(name, phase, msg) →             │
  │     loop.call_soon_threadsafe(queue.put_nowait)      │
  │                                                     │
  │   最终结果 → queue.put({"type":"result","data":...}) │
  │   哨兵 None → queue.put(None)                        │
  └─────────────────────────────────────────────────────┘
  │
  ├── SSE Generator 从 queue 读取事件
  │   ├── event: status → 前端更新 Agent 面板
  │   ├── event: result → 自动保存笔记→前端渲染→结束
  │   └── event: error  → 前端显示错误
  │
  └── StreamingResponse(media_type="text/event-stream")
```

### 3.3 下载流

```
POST /api/download_note {content, style, svg_markup?, title}
  │
  ├── 参数校验: content 非空；mindmap 需携 svg_markup
  │
  ├── 分支:
  │   ├── outline / exam_points → _generate_docx()
  │   │   ├── python-docx 创建文档
  │   │   ├── 封面(标题+副标题+分隔线)
  │   │   ├── Markdown→docx 段落渲染
  │   │   │   ├── #/##/### → Heading 1/2/3
  │   │   │   ├── **bold** / *italic* → 内联格式
  │   │   │   ├── 【术语】→ 紫色加粗
  │   │   │   └── 普通段落 → 正文
  │   │   ├── 自动生成页脚
  │   │   └── 文件名: {title}-{提纲|考点清单}.docx
  │   │
  │   └── mindmap → _generate_pdf_from_svg()
  │       ├── svglib.svg2rlg() 解析前端 SVG
  │       ├── reportlab 渲染为 PDF
  │       ├── 异常时→fallback PDF (纯文本提示)
  │       └── 文件名: {title}-思维导图.pdf
  │
  └── Response: RFC 5987 编码 Content-Disposition 头
```

---

## 4. 前端设计

### 4.1 页面结构

```
app-layout
├── sidebar (固定 220px)
│   ├── sidebar-brand (Logo + 系统名)
│   ├── sidebar-nav (4 个导航项)
│   └── sidebar-footer (版权)
└── main-wrapper (flex:1)
    ├── topbar (面包屑导航)
    └── content-area
        ├── #uploadSection    (Step 1: 文件上传卡片)
        ├── #generateSection  (Step 2: 笔记生成卡片)
        └── #resultSection    (Step 3: 结果展示卡片)
```

### 4.2 导航系统

侧边栏 4 个导航项，通过 `data-nav` 属性标记：

| nav 值 | 目标 | 说明 |
|--------|------|------|
| `home` | `#generateSection` | 主页（默认展示笔记生成） |
| `upload` | `#uploadSection` | 文件上传 |
| `result` | `#resultSection` | 结果查看 |
| `settings` | `#` | 设置（预留，未实现） |

导航切换通过 `navigateTo(name)` 实现：隐藏所有卡片，显示目标卡片，更新面包屑，更新导航项 active 状态。

### 4.3 文件上传模块

#### UI 组件

- **拖拽区** (`#uploadZone`)：点击触发文件选择，支持拖拽
- **文件信息栏** (`#fileInfo`)：显示已选文件名、大小、清除按钮
- **上传按钮** (`#btnUpload`)：文件选中后启用
- **状态区** (`#uploadStatus`)：上传中/成功/失败提示
- **文件列表** (`#kbFileList`)：知识库已有文件，每项显示文件名、大小、向量数、删除按钮

#### 交互流程

1. 点击 `/` 拖拽 → `handleFileSelect(file)` → `validateFile()` → 显示文件信息
2. 点"上传并入库" → `POST /api/upload_file` (FormData) → `fetchWithTimeout()`
3. 成功 → 状态提示 + 自动刷新文件列表 → 清除选择
4. 失败 → 状态提示错误信息

### 4.4 笔记生成模块

#### UI 组件

- **查询输入** (`#queryInput`)：500 字符限制，实时字数统计，超 450 字警告
- **风格选择** (`#styleSelect`)：提纲 / 思维导图 / 考点清单
- **生成按钮** (`#btnGenerate`)：loading 状态禁用
- **Agent 状态面板** (`#agentStatusPanel`)：生成时显示，三列实时更新
- **知识库文档列表** (`#kbDocList`)：方便在生成前查看可用素材

#### SSE 解析逻辑

```javascript
// 读取 ReadableStream，按 \n\n 分割事件
// 解析 event: status → handleAgentStatus()
// 解析 event: result → renderResult()
// 解析 event: error  → show error
```

### 4.5 结果展示模块

#### 三标签页切换

| 标签 | ID | 内容 | 样式 |
|------|-----|------|------|
| 原文素材 | `#panelContext` | 检索出素材的摘要 | 灰色背景 |
| 知识框架 | `#panelFramework` | 层级大纲框架 | 紫色背景 |
| 最终笔记 | `#panelNote` | 笔记文本 / 思维导图 SVG | 绿色背景 |

#### 思维导图渲染 (`renderMindmapToSvg()`)

纯 JS 实现的水平树状布局算法：

```
输入：Markdown 层级标题（# → ## → ### → ####）
  1. parseMarkdownToTree() → 构建树结构
  2. calculateLayout()     → 递归计算节点坐标
     - 水平间距: 200px (一级), 180px (二级+)
     - 垂直间距: 50px (同级节点)
     - 子节点垂直居中于父节点
  3. drawTree()            → 生成 SVG 元素
     - 圆角矩形节点：不同层级不同颜色
       - 中心主题 (#): 深紫 #5B21B6, 白字
       - 一级分支 (##): 紫色 #7C3AED, 白字
       - 二级分支 (###): 浅紫 #A78BFA, 深字
       - 三级分支 (####): 灰紫 #DDD6FE, 深字
     - 贝塞尔曲线连线（二次贝塞尔 cQ）
     - SVG drop-shadow 滤镜
  4. 缩放/拖拽控制:
     - 鼠标滚轮缩放 (0.5x ~ 3.0x, 步进 0.1)
     - 按钮 +/- 缩放
     - mousedown/move/up 拖拽平移
     - 缩放中心 = 鼠标位置
```

### 4.6 CSS 设计系统

#### 设计令牌（CSS Custom Properties）

- 主色：`--primary: #7C3AED`、`--primary-dark: #5B21B6`
- 灰阶：`--gray-50` ~ `--gray-900`
- 间距：`--space-xxs` ~ `--space-xl`
- 阴影：`--shadow-sm` ~ `--shadow-xl`
- 圆角：`--radius-sm: 6px`, `--radius-md: 10px`, `--radius-lg: 16px`

#### 响应式断点

```css
@media (max-width: 768px)  → sidebar 收窄, 卡片 padding 减小
@media (max-width: 480px)  → sidebar 隐藏, 单列布局
```

---

## 5. 后端模块设计

### 5.1 模块依赖图

```
main.py
├── config.py          (全局配置，最先加载)
├── schemas.py         (Pydantic 模型，无依赖)
├── agents.py          (三 Agent + LangGraph 工作流)
│   ├── config.py      (API Key / LLM 参数)
│   └── vector_store.py (向量检索)
├── vector_store.py    (ChromaDB 管理)
│   └── config.py      (路径 / 集合名 / Top-K)
├── file_handler.py    (文件处理)
│   └── config.py      (扩展名 / 大小限制 / 存储路径)
├── notes_store.py     (笔记历史)
│   └── config.py      (KNOWLEDGE_DIR)
└── __init__.py        (包文档 + 版本号)
```

**无循环依赖**：config.py 不导入任何业务模块，形成单向依赖树。

### 5.2 config.py — 全局配置中心

**加载顺序**：
1. `python-dotenv` 加载 `.env`（存在则加载，失败不阻断）
2. 模块级常量定义（路径 / 限制 / API / CORS / 日志 / 向量库）
3. 启动初始化：`mkdir -p KNOWLEDGE_DIR CHROMA_DIR LOG_DIR`

**设计原则**：
- `.env` 文件可选（所有配置有默认值）
- `dotenv` 可选（未安装时读系统环境变量）
- 真实密钥永不出现在代码中（`.env.example` 仅含占位符）

### 5.3 main.py — FastAPI 入口

#### 启动流程

```
1. setup_logging()         → 日志系统就绪
2. app = FastAPI(...)      → 创建实例
3. CORS 中间件注册          → 跨域配置
4. RequestLogging 中间件    → 请求计时
5. GlobalExceptionHandler   → 统一错误
6. 路由注册（8 个 API）
7. _ensure_services()      → 预热 FileHandler + VectorStoreManager
```

#### API 路由一览

| 路由 | 函数 | HTTP | 响应类型 |
|------|------|------|----------|
| `/api/health` | `health_check()` | GET | JSON |
| `/api/upload_file` | `api_upload_file()` | POST | JSON (UploadFileResponse) |
| `/api/generate_note` | `api_generate_note()` | POST | SSE Stream |
| `/api/download_note` | `api_download_note()` | POST | Binary (docx/pdf) |
| `/api/files` | `api_list_files()` | GET | JSON |
| `/api/files/{filename}` | `api_delete_file()` | DELETE | JSON |
| `/api/files` | `api_clear_files()` | DELETE | JSON |
| `/api/notes` | `api_list_notes()` | GET | JSON |
| `/api/notes/{note_id}` | `api_get_note()` | GET | JSON |
| `/api/notes/{note_id}` | `api_delete_note()` | DELETE | JSON |

### 5.4 schemas.py — 数据契约

```
GenerateNoteRequest
  ├── query: str (1-500, 必填)
  └── style: str (outline|mindmap|exam_points, 默认outline)

GenerateNoteResponse
  ├── code: int (默认200)
  ├── msg: str
  ├── context: str (检索素材)
  ├── framework: str (知识框架)
  ├── note: str (最终笔记)
  ├── style: str
  └── elapsed: float (耗时秒数)

DownloadNoteRequest
  ├── content: str (1-50000)
  ├── style: str (outline|mindmap|exam_points)
  ├── svg_markup: str (0-200000, mindmap专用)
  └── title: str (1-200, 默认"学习笔记")

UploadFileResponse
  ├── code, msg, filename, chunks_count

ErrorResponse
  ├── code: int (HTTP 状态码)
  ├── msg: str (人类可读)
  ├── detail: str | None (调试用)
  └── error_type: str | None (异常类型名)
```

### 5.5 notes_store.py — 笔记历史

#### 存储格式

```json
[
  {
    "id": "a1b2c3d4e5f6",
    "query": "什么是RAG？",
    "style": "outline",
    "context": "...",
    "framework": "...",
    "note": "一、RAG 概述\n...",
    "created_at": "2026-06-15 14:30:00"
  }
]
```

#### 关键行为

| 操作 | 行为 |
|------|------|
| **保存** | 相同 query+style → 覆盖旧记录（保留原 ID）；否则 → 插入到列表头部 |
| **去重** | 基于 query + style 两个字段，不含 created_at |
| **上限** | 最大 100 条，超出后删除最旧的 |
| **原子写** | 先写 `.tmp` 文件，再 `rename` 覆盖原文件 |
| **线程安全** | `threading.Lock()` 保护所有读写操作 |
| **损坏恢复** | 文件不存在/JSON 解析失败 → 返回空列表，下次保存时自动修复 |

---

## 6. 多 Agent 工作流设计

### 6.1 设计哲学

```
┌────────────────────────────────────────────────────────┐
│                   核心原则                               │
│                                                        │
│   Agent (Python 代码)  = 决策者 + 规则控制器 + 自检调度  │
│   LLM  (DeepSeek API) = 纯执行工具，仅负责文本生成      │
│                                                        │
│   ❌ LLM 不参与: 决策、自检、重试、流程跳转              │
│   ✅ 所有 if/else/retry 逻辑 = Python 原生代码          │
└────────────────────────────────────────────────────────┘
```

**为什么这样设计？**

1. **确定性**：Python 代码的行为可预测、可调试、可单元测试
2. **成本控制**：减少不必要的 LLM 调用（如"结果好不好？"这种二次评估）
3. **速度**：Python 自检是毫秒级的，LLM 自检需要数秒
4. **可解释性**：每个决策点都有明确的日志

### 6.2 LangGraph 工作流

#### 状态定义

```python
class AgentState(TypedDict, total=False):
    query: str         # 用户查询
    context: str       # LLM 摘要后的素材（供后续 LLM 使用）
    raw_context: str   # 原始检索素材（供 Python 溯源校验使用）
    framework: str     # 知识框架
    note: str          # 最终笔记
    style: str         # outline | mindmap | exam_points
    error: str         # 非空时 = 异常终止
    status: str        # running | completed | terminated
```

#### 状态图

```
START
  │
  ▼
RetrievalAgent ──── status==terminated ────▶ END (返回错误)
  │
  │ (正常)
  ▼
UnderstandingAgent ─ status==terminated ───▶ END (返回错误)
  │
  │ (正常)
  ▼
NoteGenerateAgent ── status==terminated ───▶ END (返回错误)
  │
  │ (正常)
  ▼
 END (返回完整结果)
```

条件边的路由逻辑：
- 检查 `state["status"]` 是否为 `"terminated"`
- 是 → 跳过后续节点，直接 `END`
- 否 → 继续下一节点

### 6.3 RetrievalAgent 设计

#### 职责边界

- **负责**：输入校验、向量检索、相关性判断、重试决策、摘要提纯
- **不负责**：知识框架构建、笔记格式、最终输出质量

#### 详细流程

```
Step 1: _validate_query(query)
  ├── 检查非空
  ├── 检查最小长度 (≥2 字符)
  └── 不通过 → 终止

Step 2: _do_search(query, top_k=5)
  ├── vector_store.retrieve_similar(query, top_k)
  └── 返回: [{content, source, distance}, ...]

Step 3: _check_relevance(raw_results)
  ├── 过滤条件 1: distance ≤ 0.8
  ├── 过滤条件 2: _keyword_overlap_check(query, result.content)
  │   └── bigram 关键词重合度 ≥ 30%
  ├── 通过条数 < MIN_VALID_RESULTS (1) →
  │   └── 二次检索: _do_search(query, top_k=5, threshold=0.8*1.25=1.0)
  └── 仍不足 → 终止 (返回友好提示)

Step 4: _summarize_context(valid_results) [LLM 调用]
  ├── System: "你是知识检索助手，请对以下素材做摘要提纯"
  ├── User: 拼接的有效素材
  └── 约束: ≤800 字符，保留关键术语
```

#### 相关性距离阈值设计

```
ChromaDB L2² 距离与余弦相似度关系:
  L2² = 2 - 2*cos_sim

映射表:
  distance=0.0  → cos_sim=1.0  (完全相同)
  distance=0.6  → cos_sim=0.7  (高度相关)
  distance=0.8  → cos_sim=0.6  (中等相关) ← 默认阈值
  distance=1.0  → cos_sim=0.5  (弱相关)   ← 二次检索阈值
  distance=2.0  → cos_sim=0.0  (无关)
```

### 6.4 UnderstandingAgent 设计

#### 职责边界

- **负责**：拦截空素材、调用 LLM 构建框架、溯源校验、结构合规检查
- **不负责**：初始检索、笔记最终格式

#### 溯源校验算法（5 级匹配）

```
对框架中每个【关键术语】执行:

Level 1: 精确匹配
  └── term in raw_context → ✅

Level 2: 子串匹配
  └── any(part in raw_context for part in term_parts) → ✅

Level 3: 关键词匹配
  └── any(keyword in raw_context for keyword in extract_keywords(term)) → ✅

Level 4: 字符重叠率
  └── overlap_ratio(term_chars, raw_context_chars) ≥ 50% → ✅

Level 5: 正则放宽匹配
  └── 去除【】标记后，\b{term}\b in raw_context → ✅

全部 Level 失败 → ❌ 标记为"疑似编造"
```

#### 框架合规参数

```python
MIN_SECTIONS: int = 2   # 一级节点最少数量
MIN_DEPTH: int = 2      # 最少层级深度（如 # 和 ##）
MAX_RETRIES: int = 1    # 失败重试次数
```

### 6.5 NoteGenerateAgent 设计

#### 职责边界

- **负责**：套用模板生成笔记、格式合规校验、内容降噪、重试/降级
- **不负责**：素材检索、框架构建

#### 三种风格模板

| 风格 | 模板要点 | 格式校验 | 输出特征 |
|------|----------|----------|----------|
| `outline` | 层级编号（一、(一) 1. (1) a.），【】标术语，附术语表 | `[一二三四...]+、` + `【.+?】`，≥2 节，≥200 字 | 适合系统性学习 |
| `mindmap` | Markdown 层级标题 `# → ## → ### → ####`，每行一个节点，≤20 字/节点 | `^#\s[^#]` + `^##\s`，≥3 节，≥80 字 | 适合前端 SVG 渲染 |
| `exam_points` | `考点 N：` 开头，含【核心概念】/【易错点】/【记忆口诀】，★ 考频 | `考点\s*\d+` + `【核心概念】` + `【易错点】`，≥1 节，≥300 字 | 适合备考 |

#### 内容降噪

```python
# 检测 AI 自述标记
NOISE_PATTERNS = [
    r"作为一个AI",
    r"作为一个人工智能",
    r"我是AI",
    r"根据素材",
    r"综上所述",
]

# 检测重复率
重复率 = 重复 n-gram 数量 / 总 n-gram 数量
拒绝阈值: > 30%
```

#### 重试与降级策略

```
第 1 次生成 → 格式/内容自检
  ├── 通过 → 返回
  └── 不通过 → 第 2 次生成 (temperature += 0.1)
       ├── 通过 → 返回
       └── 不通过 → 第 3 次生成 (temperature += 0.2)
            ├── 通过 → 返回
            └── 不通过 → 返回降级版笔记 (原始输出 + 警告)
```

### 6.6 SSE 实时通知协议

```python
# 状态事件（每个 Agent 工作步骤触发一次）
event: status
data: {
  "type": "status",
  "agent": "RetrievalAgent",
  "phase": "progress",     # start | progress | retry | done | error
  "message": "已检索到 5 条候选素材，正在校验相关性..."
}

# 结果事件（工作流完成时触发）
event: result
data: {
  "context": "RAG（检索增强生成）是...",
  "framework": "一、RAG 概述\n  1. 定义\n  2. 工作流程\n...",
  "note": "一、RAG 概述\n(一) 定义\n...",
  "style": "outline",
  "status": "completed",
  "elapsed_seconds": 12.5
}

# 错误事件（工作流异常时触发）
event: error
data: {"type": "error", "message": "Agent 工作流执行失败：..."}
```

---

## 7. 向量库设计

### 7.1 Embedding 后端策略

```
优先级顺序:
  1. HuggingFace 本地模型 (BAAI/bge-small-zh-v1.5, 512 维)
     ├── 优势: 离线可用、无限调用、中文优化
     ├── 大小: ~100MB
     └── 条件: langchain-huggingface + sentence-transformers 已安装

  2. OpenAI 兼容 Embedding (DeepSeek Embed API)
     ├── 优势: 无需本地存储模型
     └── 条件: DEEPSEEK_API_KEY 已配置 + 网络可达

  3. Dummy Embedding (零向量)
     ├── 优势: 永远可用
     └── 限制: 无语义区分能力，检索结果随机
```

### 7.2 ChromaDB 数据结构

```
Collection: "knowledge"
  │
  ├── id: "{batch_uuid}_{source_name}_{chunk_index}"
  │   例: "a1b2c3_机器学习入门.pdf_5"
  │
  ├── document: "这是第 5 个文本分片的内容..."
  │
  ├── embedding: [0.023, -0.154, ...]  (512 维 float32)
  │
  └── metadata:
      ├── source: "机器学习入门.pdf"
      ├── chunk_index: 5
      └── batch_id: "a1b2c3"
```

### 7.3 增量更新策略

```
add_documents(chunks, source_name):
  1. 生成 batch_uuid = uuid4().hex[:8]
  2. 逐片向量化 + 构建 Chroma Document 列表
  3. collection.add(ids, documents, embeddings, metadatas)
  4. 不删除已有数据（纯追加）
  5. 同一文件重新上传 = 新 batch_uuid，旧数据仍保留
```

### 7.4 检索接口

```python
retrieve_similar(query, top_k=5) → List[Dict]:
  [
    {
      "content": "分片文本内容...",
      "source": "机器学习入门.pdf",
      "chunk_index": 3,
      "distance": 0.42  # L2² 距离，越小越相关
    }
  ]
```

---

## 8. 文件处理管线

### 8.1 多格式文本提取

#### TXT 提取

```
尝试 UTF-8 解码 → 成功: 返回
                → 失败: 尝试 GBK   → 成功: 返回
                                  → 失败: 尝试 Latin-1 (保底)
```

#### Markdown 提取

```
读取 UTF-8 文本
  → markdown.markdown() 转 HTML
  → 正则 <[^>]+> 剥离 HTML 标签
  → 返回纯文本
异常 → 返回原始文本 (当作 TXT 处理)
```

#### PDF 提取

```
PdfReader(file)
  → 检查是否加密 (reader.is_encrypted → decrypt)
  → 逐页 extract_text()
  → 过滤空页
  → 拼接返回
```

### 8.2 文本清洗四阶段

```
输入: 原始文本

Phase 1: Unicode 标准化
  └── unicodedata.normalize('NFKC', text)

Phase 2: 字符清理
  ├── 移除控制字符 (保留 \n \t)
  ├── 移除 Unicode 私有区字符
  ├── 移除零宽/不可见字符
  └── 移除 Unicode 替换字符 "�"

Phase 3: 空白标准化
  ├── 连续空格/Tab → 单个空格
  └── 连续 3+ 换行 → 2 个换行

Phase 4: 修剪
  └── strip() 首尾空白

输出: 干净文本
```

### 8.3 文本分片参数

```
RecursiveCharacterTextSplitter:
  chunk_size: 1000 字符
  chunk_overlap: 200 字符

分隔符优先级（中文优化）:
  ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
```

---

## 9. 笔记下载设计

### 9.1 DOCX 生成（提纲 / 考点清单）

#### 文档结构

```
[封面]
  ├── 留白
  ├── 标题 (Title 样式, 22pt, 深紫 #4F1EA3, 居中)
  ├── 副标题 (12pt, 灰色 #999999, 斜体, 居中)
  ├── 分隔线
  └── 留白

[正文] — Markdown 逐段渲染
  ├── # 标题   → Heading 1 (16pt, #333)
  ├── ## 标题  → Heading 2 (13pt, #555)
  ├── ### 标题 → Heading 3 (11pt, #777)
  ├── **粗体** → Bold
  ├── *斜体*   → Italic
  ├── 【术语】 → Bold + 紫色 #7C3AED
  └── 普通段落 → Normal (10.5pt, #444)

[页脚]
  └── "— 由本地知识库学习笔记生成助手自动生成 —" (8pt, 灰色, 斜体, 居中)
```

#### 页面设置

```python
section.top_margin    = 2.5 cm
section.bottom_margin = 2.5 cm
section.left_margin   = 2.8 cm
section.right_margin  = 2.8 cm
```

### 9.2 PDF 生成（思维导图）

#### 主流程

```
前端 SVG (字符串) → svglib.svg2rlg() → ReportLab Drawing
  → renderPDF.draw() → canvas → BytesIO → PDF bytes
```

#### Fallback 策略

```
尝试 svglib 解析 SVG
  ├── 成功 → 生成含 SVG 的 PDF
  └── 例外 (AttributeError / 解析失败)
      └── Fallback PDF: 纯文本提示 + 建议使用浏览器截图
```

### 9.3 RFC 5987 文件名编码

HTTP `Content-Disposition` 头仅支持 Latin-1 编码。含中文的文件名需要 RFC 5987 编码：

```python
from urllib.parse import quote
encoded = quote("什么是RAG-提纲.docx", safe="")
# → "%E4%BB%80%E4%B9%88%E6%98%AFRAG-%E6%8F%90%E7%BA%B2.docx"

header = f"attachment; filename*=UTF-8''{encoded}"
```

前端解析：
```javascript
var match = disposition.match(/filename\*=UTF-8''(.+)/);
downloadName = decodeURIComponent(match[1]);
```

---

## 10. API 协议规范

### 10.1 统一成功响应

```json
{
  "code": 200,
  "msg": "操作描述",
  "...": "业务字段"
}
```

### 10.2 统一错误响应

```json
{
  "code": 400,       // 或 500
  "msg": "人类可读错误描述",
  "detail": null,     // DEBUG_MODE=true 时含详细堆栈
  "error_type": null  // DEBUG_MODE=true 时含异常类名
}
```

### 10.3 HTTP 状态码约定

| 状态码 | 含义 | 示例场景 |
|--------|------|----------|
| 200 | 成功 | 请求正常处理 |
| 400 | 客户端错误 | 参数为空、文件格式不支持、query 超长 |
| 500 | 服务器内部错误 | LLM 调用失败、向量库写入异常、文件系统错误 |

### 10.4 超时与重试

| 层级 | 超时 | 说明 |
|------|------|------|
| 前端 fetch | 360s (6 分钟) | AbortController + setTimeout |
| LLM 调用 | 300s | 通过 `request_timeout` 参数 |
| Agent 内部重试 | 同步循环 | Retrieval: 1 次放宽, Understanding: 1 次重试, Generation: 2 次重试 |

---

## 11. 前端 SSE 流处理

### 11.1 事件解析

```javascript
async function fetchSSE(url, body) {
  const response = await fetch(url, { method: 'POST', body: JSON.stringify(body) });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // 按 \n\n 分割事件
    const parts = buffer.split('\n\n');
    buffer = parts.pop(); // 保留最后一个不完整片段

    for (const part of parts) {
      const lines = part.split('\n');
      let eventType = '', data = '';
      for (const line of lines) {
        if (line.startsWith('event: ')) eventType = line.slice(7);
        else if (line.startsWith('data: ')) data = line.slice(6);
      }
      if (eventType && data) {
        const parsed = JSON.parse(data);
        handleSSEEvent(eventType, parsed);
      }
    }
  }
}
```

### 11.2 Agent 状态更新

前端根据 `event: status` 更新三个 Agent 的 UI 状态：

```javascript
{
  "RetrievalAgent":   { icon: "🔍", name: "检索Agent" },
  "UnderstandingAgent": { icon: "🧩", name: "理解Agent" },
  "NoteGenerateAgent": { icon: "✍️", name: "生成Agent" }
}

状态映射:
  "start"    → 蓝色边框 + 旋转动画 + "运行中" badge
  "progress" → 保持蓝色 + 更新状态文本
  "retry"    → 橙色边框 + "重试中" badge + 更新状态文本
  "done"     → 绿色边框 + "✅" badge
  "error"    → 红色边框 + "⚠️" badge + 显示错误文本
```

---

## 12. 状态管理

### 12.1 后端无状态设计

FastAPI 应用本身无会话状态。所有状态外存：

| 状态 | 存储位置 | 生命周期 |
|------|----------|----------|
| 向量索引 | `chroma_db/` | 永久 |
| 原始文件 | `knowledge/` | 永久（手动删除） |
| 笔记历史 | `knowledge/notes_history.json` | 永久（最多 100 条） |
| 应用日志 | `logs/app.log` | 轮转（10MB × 5 备份） |
| LLM 实例 | 模块变量 | 进程生命周期 |

### 12.2 前端单页状态

前端是纯单页应用（SPA），无路由库，所有状态靠 DOM 切换：

- **当前页面**：哪个 section 可见
- **上传状态**：`uploadStatus.textContent`
- **生成状态**：`generateStatus.textContent` + Agent 面板
- **结果数据**：`resultMeta.textContent` + Tab 面板内容
- **思维导图**：`mindmapContainer` 的 display + zoomLevel + SVG 内容

---

## 13. 错误处理策略

### 13.1 分层错误处理

```
┌──────────────────────────────────────────────┐
│ 前端                                           │
│  • fetchWithTimeout (360s)                     │
│  • HTTP 状态码检查 (response.ok)               │
│  • SSE error 事件捕获                          │
│  • 网络异常 catch (response 不存在)             │
├──────────────────────────────────────────────┤
│ 路由层 (main.py)                               │
│  • Pydantic 请求校验 (422 → FastAPI 自动)      │
│  • 二次手动校验 (400 → ErrorResponse)           │
│  • 每个步骤 try/except → 返回 500              │
├──────────────────────────────────────────────┤
│ 中间件层                                       │
│  • GlobalExceptionHandler → 统一 500           │
│  • DEBUG_MODE=true → 附带 detail + error_type  │
├──────────────────────────────────────────────┤
│ Agent 层                                       │
│  • Agent 内部自检失败 → 重试 → 降级 → 终止     │
│  • LLM 调用异常 → RuntimeError → 终止          │
│  • 工作流线程内异常 → error SSE 事件            │
├──────────────────────────────────────────────┤
│ 服务层                                         │
│  • FileHandler: 明确 ValueError / IOError      │
│  • VectorStore: 检查依赖可用性                  │
│  • notes_store: 文件损坏 → 自动修复             │
└──────────────────────────────────────────────┘
```

### 13.2 优雅降级路径

| 场景 | 降级策略 |
|------|----------|
| HuggingFace 模型不可用 | → Dummy Embedding (向量检索无语义区分) |
| dotenv 未安装 | → 使用系统环境变量 |
| LangChain 版本不兼容 | → 多重 import 路径 fallback |
| LLM API Key 为占位符 | → RuntimeError + 明确提示配置 .env |
| Agent 重试耗尽 | → 返回降级版笔记或友好终止消息 |
| 笔记自动保存失败 | → 不影响 SSE 响应，仅记 warning 日志 |
| PDF SVG 解析失败 | → Fallback PDF（纯文本提示） |
| 日志文件不可写 | → 仅控制台输出，不阻断服务 |
| 知识库目录创建失败 | → 静默跳过，后续接口自行报错 |

---

## 14. 安全考量

### 14.1 输入安全

| 威胁 | 防护 |
|------|------|
| 路径穿越 | 文件名净化（移除 `../`、`..\\`、`/`、`\\`） |
| 超大文件 | MAX_FILE_SIZE=20MB 限制 |
| 恶意文件类型 | 仅允许 `{txt, pdf, md}` 扩展名白名单 |
| XSS（前端） | 文件列表 `escapeHtml()`，不使用 innerHTML + 用户数据 |
| query 注入 | Pydantic max_length=500 限制 |

### 14.2 输出安全

| 威胁 | 防护 |
|------|------|
| LLM 幻觉 | 溯源校验（5 级匹配）+ 内容降噪 + 格式校验 |
| 敏感信息泄露 | `.env` 已在 `.gitignore`，`.env.example` 仅含占位符 |
| 调试信息泄露 | `DEBUG_MODE=false` 时不返回 `detail` 和 `error_type` |
| 内部路径暴露 | 所有响应不包含服务器文件路径 |

### 14.3 API 安全

| 措施 | 说明 |
|------|------|
| CORS 白名单 | 仅允许配置的域名，默认 localhost |
| Swagger/ReDoc | 仅在 `DEBUG_MODE=true` 时开启 |
| 请求日志 | 记录所有请求的 method/path/status/duration，不含 body |

---

## 15. 环境与部署

### 15.1 运行环境

```
Python:     ≥ 3.11
OS:         Windows 10/11, macOS 12+, Linux
Memory:     ≥ 2GB (ChromaDB + Embedding 模型)
Disk:       ≥ 500MB (模型 ~100MB + ChromaDB + 日志)
Network:    需要访问 api.deepseek.com (LLM 调用)
            不需要外部访问 (本机 localhost)
```

### 15.2 部署模式

#### 开发模式

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- `--reload`：代码变更自动重启
- `DEBUG_MODE=true`：开启 Swagger + 详细错误信息

#### 生产模式

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

- `DEBUG_MODE=false`：关闭 API 文档 + 隐藏调试信息
- `--workers 2`：多 worker（注意：多 worker 下 ChromaDB 需配置为 HTTP 模式）

#### 前端部署

前端是纯静态文件，可部署到：
- 本地直接用浏览器打开 `index.html`
- Vercel / Netlify / GitHub Pages
- Nginx 静态文件服务（需配置 `CORS_ALLOW_ORIGINS`）

### 15.3 环境变量完整清单

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `DEEPSEEK_API_KEY` | ✅ | — | DeepSeek API 密钥 |
| `DEEPSEEK_API_BASE` | ❌ | `https://api.deepseek.com/v1` | API 端点 |
| `DEEPSEEK_CHAT_MODEL` | ❌ | `deepseek-chat` | 对话模型 |
| `DEEPSEEK_EMBED_MODEL` | ❌ | `deepseek-embed` | Embedding 模型 |
| `LLM_TEMPERATURE` | ❌ | `0.7` | 生成温度 |
| `LLM_MAX_TOKENS` | ❌ | `8192` | 最大输出 |
| `LLM_REQUEST_TIMEOUT` | ❌ | `300` | LLM 超时(秒) |
| `SERVER_HOST` | ❌ | `0.0.0.0` | 监听地址 |
| `SERVER_PORT` | ❌ | `8000` | 监听端口 |
| `DEBUG_MODE` | ❌ | `false` | 调试模式 |
| `LOG_LEVEL` | ❌ | `INFO` | 日志级别 |
| `CORS_ALLOW_ORIGINS` | ❌ | `localhost:3000,...` | 跨域白名单 |
| `VECTOR_COLLECTION_NAME` | ❌ | `knowledge` | 向量集合名 |
| `VECTOR_SEARCH_TOP_K` | ❌ | `5` | 检索 Top-K |

---

## 16. 关键技术决策

### 决策 1：为什么用三 Agent 串行而非单 Agent 端到端？

**决策**：将笔记生成拆分为检索→理解→生成三个职能 Agent，各司其职。

**理由**：
- **可解释性**：每个阶段产出可见（context / framework / note），问题定位清晰
- **可控性**：每个 Agent 有自己的自检逻辑，不合适时可以终止或重试单个阶段
- **质量**：阶段化处理让 LLM 能聚焦单一任务，比端到端一次性生成更准确
- **复用**：context 和 framework 可以独立展示给用户（三标签页功能）

### 决策 2：为什么 LLM 不参与决策？

**决策**：所有 if/else/retry/路由 逻辑用纯 Python 实现，LLM 仅调用 `_call_llm()` 生成文本。

**理由**：
- **确定性**：规则检验（如正则匹配、距离阈值）不会随机失败
- **可测试**：Python 逻辑可以写单元测试，LLM 输出不能
- **成本**：减少不必要 LLM 调用（如"这个输出好不好？"类二次评估）
- **速度**：Python 校验毫秒级，LLM 校验秒级
- **安全性**：避免 prompt injection 影响流程控制

### 决策 3：为什么选择 BAAI/bge-small-zh-v1.5 做 Embedding？

**决策**：使用本地 HuggingFace 模型而非远程 API。

**理由**：
- **隐私**：文档内容不上传到外部 Embedding 服务
- **成本**：无限调用，无 API 费用
- **离线**：不依赖网络连接
- **中文优化**：BGE 系列在中文语义理解上表现优秀
- **轻量**：仅 ~100MB 下载，CPU 推理即可

### 决策 4：为什么不用 WebSockets 而用 SSE？

**决策**：笔记生成用 Server-Sent Events 推送进度。

**理由**：
- **单向性**：状态推送是纯服务端→客户端，不需要双向通信
- **简单**：SSE 是 HTTP 原生支持的，不需要额外协议升级
- **自动重连**：浏览器原生 EventSource 支持断线重连
- **兼容性**：Nginx 等反向代理默认支持 HTTP 流
- **够用**：4 种事件类型（status / result / error / 哨兵）完全满足需求

### 决策 5：为什么用 JSON 文件而非数据库存笔记历史？

**决策**：用 `notes_history.json` 单文件存储笔记历史。

**理由**：
- **量级**：个人工具，最多 100 条，JSON 文件毫秒级读写
- **零依赖**：不需要引入 SQLite / PostgreSQL 依赖
- **可读**：JSON 格式人类可读，方便调试和手动编辑
- **便携**：单文件即可迁移全部历史
- **原子性**：tmp + rename 保证写入不损坏

### 决策 6：为什么思维导图是前端渲染而非后端生成图片？

**决策**：思维导图渲染在浏览器端完成。

**理由**：
- **交互性**：前端 SVG 支持实时缩放、拖拽，用户体验好
- **渲染质量**：浏览器 SVG 渲染比后端图形库更精确
- **减少后端负载**：不占用服务器资源做图形计算
- **按需下载**：用户浏览满意后再点下载转 PDF

---

## 附录 A：文件大小统计

| 文件 | 行数 | 大小 | 说明 |
|------|------|------|------|
| `main.py` | 1607 | ~55KB | 最大单体文件（含完整文档生成逻辑） |
| `agents.py` | 1797 | ~62KB | Agent 定义 + 工作流编排 |
| `file_handler.py` | 821 | ~28KB | 文件处理 |
| `vector_store.py` | 780 | ~27KB | 向量库管理 |
| `script.js` | 1637 | ~58KB | 前端全逻辑 |
| `styles.css` | 1296 | ~32KB | 前端全样式 |
| `schemas.py` | 150 | ~4.5KB | 数据模型 |
| `config.py` | 169 | ~5.5KB | 全局配置 |
| `notes_store.py` | 271 | ~8KB | 笔记历史 |
| `index.html` | 337 | ~10KB | 页面结构 |

---

## 附录 B：项目演化记录

| 日期 | 事件 | 影响 |
|------|------|------|
| 2026-06 | v0.1.0 初始版本 | 完整的三大模块（上传/生成/结果）+ 三 Agent 工作流 |
| 2026-06 | 代码优化 | 移除 1 个死文件（utils.py）、5 个未使用导入、4 个死配置常量、1 段死 CSS、1 个死字典；修复 .env.example 密钥泄露 |
| 2026-06 | 下载修复 | 修复中文文件名 RFC 5987 编码问题（UnicodeEncodeError → 使用 `urlparse.quote()`）|
| 2026-06 | 前端反馈修复 | 下载错误提示从隐藏元素迁移到可见区域 |

---

> 📝 本文档随项目同步更新。如需了解最新的架构细节，请阅读源码中的模块级 docstring，每个文件头部都有详细的职责和原则说明。
