/**
 * ============================================================
 * script.js — 本地知识库学习笔记生成助手（前端交互逻辑）
 * ============================================================
 *
 * 核心功能：
 *   1. 文件拖拽/点击上传 + 前端预校验（格式/大小）
 *   2. 调用 POST /api/upload_file → 展示上传状态
 *   3. 调用 POST /api/generate_note → 展示生成状态
 *   4. 结构化渲染结果：原文素材 / 知识框架 / 最终笔记（三栏标签切换）
 *   5. 全局异常处理：网络错误、超时、后端错误提示
 *
 * 设计原则：
 *   - 所有请求使用 fetch，处理后端 CORS 跨域
 *   - 上传/生成过程中禁止重复点击（按钮置灰 + loading 动画）
 *   - 纯原生 JS，零依赖，可直接部署至 Vercel
 * ============================================================
 */

(function () {
  "use strict";

  // ============================================================
  // 一、常量 & 配置
  // ============================================================

  /** @const {string} 后端 API 基础地址（后端已开启 CORS） */
  const API_BASE = "http://127.0.0.1:8000/api";

  /** @const {string[]} 允许的文件扩展名（小写） */
  const ALLOWED_EXTENSIONS = ["txt", "pdf", "md"];

  /** @const {number} 单文件最大字节数（20MB） */
  const MAX_FILE_SIZE = 20 * 1024 * 1024;

  /** @const {number} fetch 请求超时时间（毫秒） */
  const REQUEST_TIMEOUT = 360000; // 6 分钟（笔记生成可能较慢）

  // ============================================================
  // 二、DOM 元素引用（一次性获取，避免反复查询）
  // ============================================================

  // ---- 文件上传区域 ----
  const uploadZone = document.getElementById("uploadZone");
  const fileInput = document.getElementById("fileInput");
  const fileInfo = document.getElementById("fileInfo");
  const fileName = document.getElementById("fileName");
  const fileSize = document.getElementById("fileSize");
  const btnClearFile = document.getElementById("btnClearFile");
  const btnUpload = document.getElementById("btnUpload");
  const uploadStatus = document.getElementById("uploadStatus");

  // ---- 知识库文档列表（上传区） ----
  const kbFileList = document.getElementById("kbFileList");
  const kbEmpty = document.getElementById("kbEmpty");
  const btnRefreshFiles = document.getElementById("btnRefreshFiles");

  // ---- 知识库文档列表（主页直接可见） ----
  const kbDocList = document.getElementById("kbDocList");
  const kbDocEmpty = document.getElementById("kbDocEmpty");
  const btnRefreshKbDocs = document.getElementById("btnRefreshKbDocs");

  // ---- 笔记生成区域 ----
  const queryInput = document.getElementById("queryInput");
  const queryCharCount = document.getElementById("queryCharCount");
  const styleSelect = document.getElementById("styleSelect");
  const btnGenerate = document.getElementById("btnGenerate");
  const generateStatus = document.getElementById("generateStatus");

  // ---- 结果展示区域 ----
  const resultSection = document.getElementById("resultSection");
  const resultStyleBadge = document.getElementById("resultStyleBadge");
  const resultTabs = document.getElementById("resultTabs");
  const panelContext = document.getElementById("panelContext");
  const panelFramework = document.getElementById("panelFramework");
  const panelNote = document.getElementById("panelNote");
  const noteTextBlock = document.getElementById("noteTextBlock");
  const mindmapContainer = document.getElementById("mindmapContainer");
  const mindmapSvg = document.getElementById("mindmapSvg");
  const mindmapSvgWrapper = document.getElementById("mindmapSvgWrapper");
  const resultMeta = document.getElementById("resultMeta");
  const btnDownload = document.getElementById("btnDownload");
  const btnDownloadText = document.getElementById("btnDownloadText");

  // ---- 思维导图缩放与拖拽 ----
  const btnZoomIn = document.getElementById("btnZoomIn");
  const btnZoomOut = document.getElementById("btnZoomOut");
  const btnZoomReset = document.getElementById("btnZoomReset");
  const zoomLevelEl = document.getElementById("zoomLevel");

  // ---- Agent 状态面板 ----
  const agentStatusPanel = document.getElementById("agentStatusPanel");
  const agentStatusRetrieval = document.getElementById("agentStatusRetrieval");
  const agentStatusUnderstanding = document.getElementById("agentStatusUnderstanding");
  const agentStatusGeneration = document.getElementById("agentStatusGeneration");

  var mmZoom = 1.0;                // 当前缩放比例
  var MM_ZOOM_MIN = 0.25;          // 最小缩放
  var MM_ZOOM_MAX = 3.0;           // 最大缩放
  var MM_ZOOM_STEP = 0.15;         // 按钮缩放步长

  // 拖拽平移状态
  var mmPanX = 0;                  // 水平平移量（px）
  var mmPanY = 0;                  // 垂直平移量（px）
  var mmDragging = false;          // 是否正在拖拽
  var mmDragStartX = 0;            // 拖拽起始鼠标 X
  var mmDragStartY = 0;            // 拖拽起始鼠标 Y
  var mmDragStartPanX = 0;         // 拖拽起始平移 X
  var mmDragStartPanY = 0;         // 拖拽起始平移 Y

  // ---- 历史笔记 ----
  const historyList = document.getElementById("historyList");
  const historyEmpty = document.getElementById("historyEmpty");
  const btnRefreshNotes = document.getElementById("btnRefreshNotes");

  // ---- 侧边栏导航 ----
  const sidebarNavEl = document.querySelector(".sidebar-nav");
  const breadcrumbCurrent = document.querySelector(".breadcrumb-current");

  /** 导航配置：data-nav → 对应内容卡片 ID + 面包屑文本 */
  var NAV_MAP = {
    home:     { cardId: "generateSection", crumb: "主页" },
    upload:   { cardId: "uploadSection",   crumb: "文件上传" },
    result:   { cardId: "resultSection",   crumb: "笔记结果" },
    settings: { cardId: null,              crumb: "设置" },
  };

  /** 切换到指定导航面板 */
  function navigateTo(navKey) {
    var config = NAV_MAP[navKey];
    if (!config) return;

    // 更新导航高亮
    var allNavItems = sidebarNavEl ? sidebarNavEl.querySelectorAll(".nav-item") : [];
    for (var ni = 0; ni < allNavItems.length; ni++) {
      var it = allNavItems[ni];
      if (it.getAttribute("data-nav") === navKey) {
        it.classList.add("active");
      } else {
        it.classList.remove("active");
      }
    }

    // 更新面包屑
    if (breadcrumbCurrent) {
      breadcrumbCurrent.textContent = config.crumb;
    }

    // 显示目标卡片，隐藏其余
    var cards = document.querySelectorAll(".content-card");
    for (var ci = 0; ci < cards.length; ci++) {
      cards[ci].style.display = "none";
    }
    if (config.cardId) {
      var target = document.getElementById(config.cardId);
      if (target) {
        target.style.display = "";
      }
    }

    // 切换到笔记结果时刷新历史列表
    if (navKey === "result") {
      fetchNotesHistory();
    }
  }

  // 绑定侧边栏导航点击
  if (sidebarNavEl) {
    sidebarNavEl.addEventListener("click", function (e) {
      var navItem = e.target.closest(".nav-item");
      if (!navItem) return;
      e.preventDefault();
      var navKey = navItem.getAttribute("data-nav");
      if (navKey) navigateTo(navKey);
    });
  }

  // ============================================================
  // 三、工具函数
  // ============================================================

  /**
   * 格式化文件大小为人类可读格式
   * @param {number} bytes - 字节数
   * @returns {string} 格式化后的字符串（如 "1.5 MB"）
   */
  function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  }

  /**
   * 获取文件扩展名（小写、无点号）
   * @param {string} filename - 文件名
   * @returns {string} 扩展名
   */
  function getFileExtension(filename) {
    const parts = filename.split(".");
    if (parts.length < 2) return "";
    return parts[parts.length - 1].toLowerCase();
  }

  /**
   * 设置按钮加载状态
   * @param {HTMLButtonElement} btn - 按钮元素
   * @param {boolean} loading - 是否加载中
   * @param {string} [text] - 按钮文字（loading=false 时还原）
   */
  function setButtonLoading(btn, loading, text) {
    if (loading) {
      btn.disabled = true;
      btn.classList.add("loading");
    } else {
      btn.disabled = false;
      btn.classList.remove("loading");
      if (text) {
        const span = btn.querySelector(".btn-text");
        if (span) span.textContent = text;
      }
    }
  }

  /**
   * 设置状态提示
   * @param {HTMLElement} el - 状态元素
   * @param {"loading"|"success"|"error"|""} type - 状态类型
   * @param {string} message - 提示文字
   */
  function setStatus(el, type, message) {
    // 清除所有状态类
    el.classList.remove("loading", "success", "error");
    if (type) {
      el.classList.add(type);
    }
    el.textContent = message;
  }

  /**
   * 带超时的 fetch 封装
   * @param {string} url - 请求 URL
   * @param {RequestInit} options - fetch 选项
   * @param {number} timeout - 超时毫秒数
   * @returns {Promise<Response>}
   */
  async function fetchWithTimeout(url, options, timeout) {
    const controller = new AbortController();
    const timer = setTimeout(function () {
      controller.abort();
    }, timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });
      return response;
    } finally {
      clearTimeout(timer);
    }
  }

  // ============================================================
  // 四、前端文件校验
  // ============================================================

  /**
   * 校验文件：扩展名 + 大小
   * @param {File} file - 文件对象
   * @returns {{ valid: boolean, error: string }} 校验结果
   */
  function validateFile(file) {
    // 校验 1：扩展名
    const ext = getFileExtension(file.name);
    if (!ext || !ALLOWED_EXTENSIONS.includes(ext)) {
      return {
        valid: false,
        error: "不支持的文件格式「." + ext + "」，仅支持 " + ALLOWED_EXTENSIONS.join("、"),
      };
    }

    // 校验 2：文件大小
    if (file.size === 0) {
      return { valid: false, error: "文件内容为空，请选择有效文件。" };
    }
    if (file.size > MAX_FILE_SIZE) {
      return {
        valid: false,
        error:
          "文件过大（" +
          formatFileSize(file.size) +
          "），单文件上限为 " +
          formatFileSize(MAX_FILE_SIZE) +
          "。",
      };
    }

    return { valid: true, error: "" };
  }

  // ============================================================
  // 五、文件选择 & 拖拽处理
  // ============================================================

  /**
   * 处理文件选择（来自 input 或 拖拽）
   * @param {File} file - 选中的文件
   */
  function handleFileSelect(file) {
    const result = validateFile(file);

    if (!result.valid) {
      setStatus(uploadStatus, "error", result.error);
      clearSelectedFile();
      return;
    }

    // 显示文件信息
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    fileInfo.style.display = "flex";
    uploadZone.style.display = "none";
    btnUpload.disabled = false;
    setStatus(uploadStatus, "", "");
  }

  /** 清除已选文件 */
  function clearSelectedFile() {
    fileInput.value = "";
    fileInfo.style.display = "none";
    uploadZone.style.display = "";
    btnUpload.disabled = true;
  }

  // ---- 点击上传区域 → 触发 file input ----
  uploadZone.addEventListener("click", function () {
    fileInput.click();
  });

  // ---- file input 变化 ----
  fileInput.addEventListener("change", function () {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFileSelect(fileInput.files[0]);
    }
  });

  // ---- 清除按钮 ----
  btnClearFile.addEventListener("click", function (e) {
    e.stopPropagation(); // 防止冒泡到 uploadZone
    clearSelectedFile();
    setStatus(uploadStatus, "", "");
  });

  // ---- 拖拽支持 ----
  uploadZone.addEventListener("dragover", function (e) {
    e.preventDefault();
    uploadZone.classList.add("drag-over");
  });

  uploadZone.addEventListener("dragleave", function (e) {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");
  });

  uploadZone.addEventListener("drop", function (e) {
    e.preventDefault();
    uploadZone.classList.remove("drag-over");

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  });

  // ============================================================
  // 六、文件上传逻辑
  // ============================================================

  btnUpload.addEventListener("click", async function () {
    const file = fileInput.files[0];
    if (!file) {
      setStatus(uploadStatus, "error", "请先选择一个文件。");
      return;
    }

    // 二次校验（兜底）
    const result = validateFile(file);
    if (!result.valid) {
      setStatus(uploadStatus, "error", result.error);
      return;
    }

    // ---- 进入加载状态 ----
    setButtonLoading(btnUpload, true);
    setStatus(uploadStatus, "loading", "正在上传并入库，请稍候...");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetchWithTimeout(
        API_BASE + "/upload_file",
        { method: "POST", body: formData },
        REQUEST_TIMEOUT,
      );

      const data = await response.json();

      if (response.ok && data.code === 200) {
        // 成功
        setStatus(
          uploadStatus,
          "success",
          "上传成功！文件「" + data.filename + "」已入库（" + data.chunks_count + " 个分片）。",
        );
        // 重置文件选择，允许继续上传
        clearSelectedFile();
      } else {
        // 后端返回业务错误
        setStatus(uploadStatus, "error", data.msg || "上传失败，请重试。");
      }
    } catch (error) {
      // 网络错误 / 超时
      if (error.name === "AbortError") {
        setStatus(uploadStatus, "error", "上传请求超时，请检查网络或后端服务状态。");
      } else {
        setStatus(uploadStatus, "error", "网络请求失败，请确认后端服务已启动（http://127.0.0.1:8000）。");
      }
      console.error("[上传异常]", error);
    } finally {
      // 恢复按钮状态
      setButtonLoading(btnUpload, false, "上传并入库");
      if (!fileInput.files || fileInput.files.length === 0) {
        btnUpload.disabled = true;
      }
      // 刷新文档列表
      fetchFileList();
    }
  });

  // ============================================================
  // 六-B、知识库文档列表 & 删除
  // ============================================================

  /**
   * 获取已入库文件列表并渲染到两个位置
   */
  async function fetchFileList() {
    try {
      kbEmpty.textContent = "加载中...";
      kbEmpty.style.display = "";
      kbDocEmpty.textContent = "加载中...";
      kbDocEmpty.style.display = "";

      var response = await fetchWithTimeout(
        API_BASE + "/files",
        { method: "GET" },
        15000,
      );

      if (!response.ok) {
        kbEmpty.textContent = "加载失败，请刷新重试";
        kbDocEmpty.textContent = "加载失败，请刷新重试";
        return;
      }

      var data = await response.json();
      if (data.code === 200) {
        var files = data.files || [];
        renderFileListTo(kbFileList, kbEmpty, files);
        renderFileListTo(kbDocList, kbDocEmpty, files);
      } else {
        kbEmpty.textContent = data.msg || "加载失败";
        kbDocEmpty.textContent = data.msg || "加载失败";
      }
    } catch (error) {
      console.error("[文件列表异常]", error);
      kbEmpty.textContent = "加载失败，请确认后端服务已启动";
      kbDocEmpty.textContent = "加载失败，请确认后端服务已启动";
    }
  }

  /**
   * 渲染文件列表到指定容器
   * @param {HTMLElement} container — 文件列表容器
   * @param {HTMLElement} emptyEl   — 空状态提示元素
   * @param {Array}       files     — [{ filename, size_display, vectors_count }]
   */
  function renderFileListTo(container, emptyEl, files) {
    // 清除旧内容
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }

    if (!files || files.length === 0) {
      emptyEl.textContent = "暂无文档，请上传文件到知识库";
      emptyEl.style.display = "";
      container.appendChild(emptyEl);
      return;
    }

    emptyEl.style.display = "none";

    // 文件图标 SVG（按扩展名变色）
    var iconColors = {
      pdf: "#EF4444",
      txt: "#3B82F6",
      md: "#10B981",
    };

    for (var i = 0; i < files.length; i++) {
      var f = files[i];
      // 兼容旧格式（字符串数组）和新格式（对象数组）
      var fname = typeof f === "string" ? f : (f.filename || f.name || "unknown");
      var fsize = typeof f === "string" ? "" : (f.size_display || "");
      var fvecs = typeof f === "string" ? "" : (f.vectors_count || 0);
      var ext = fname.split(".").pop().toLowerCase();
      var iconColor = iconColors[ext] || "#6B7280";

      var item = document.createElement("div");
      item.className = "kb-file-item";

      item.innerHTML =
        '<div class="kb-file-icon">' +
          '<svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="' + iconColor + '" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">' +
            '<rect x="2.5" y="1.5" width="15" height="17" rx="1.5"/>' +
            '<line x1="6" y1="5.5" x2="14" y2="5.5"/>' +
            '<line x1="6" y1="9.5" x2="14" y2="9.5"/>' +
            '<line x1="6" y1="13.5" x2="10" y2="13.5"/>' +
          '</svg>' +
        '</div>' +
        '<div class="kb-file-info">' +
          '<div class="kb-file-name" title="' + escapeHtml(fname) + '">' + escapeHtml(fname) + '</div>' +
          '<div class="kb-file-meta">' +
            '<span>' + (fsize || "—") + '</span>' +
            '<span>' + (fvecs ? fvecs + " 个分片" : "") + '</span>' +
          '</div>' +
        '</div>' +
        '<button class="kb-file-delete" title="删除此文档" data-filename="' + escapeHtml(fname) + '">' +
          '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">' +
            '<polyline points="2.5,4.5 13.5,4.5"/>' +
            '<path d="M5.5 4.5V3a1 1 0 011-1h3a1 1 0 011 1v1.5M11.5 4.5v8a1 1 0 01-1 1h-5a1 1 0 01-1-1v-8"/>' +
          '</svg>' +
        '</button>';

      container.appendChild(item);
    }

    // 绑定删除事件
    var deleteBtns = container.querySelectorAll(".kb-file-delete");
    for (var d = 0; d < deleteBtns.length; d++) {
      deleteBtns[d].addEventListener("click", function (e) {
        e.stopPropagation();
        var fname = this.getAttribute("data-filename");
        if (fname) {
          deleteFile(fname);
        }
      });
    }
  }

  /**
   * 删除指定文件（向量库 + 物理文件）
   * @param {string} filename — 文件名
   */
  async function deleteFile(filename) {
    if (!confirm("确定要删除「" + filename + "」吗？\n此操作将同时移除向量数据和本地文件，不可恢复。")) {
      return;
    }

    try {
      var response = await fetchWithTimeout(
        API_BASE + "/files/" + encodeURIComponent(filename),
        { method: "DELETE" },
        30000,
      );

      var data = await response.json();

      if (response.ok && data.code === 200) {
        setStatus(uploadStatus, "success", data.msg || "删除成功");
        // 刷新列表
        fetchFileList();
      } else {
        setStatus(uploadStatus, "error", data.msg || "删除失败，请重试");
      }
    } catch (error) {
      console.error("[删除文件异常]", error);
      setStatus(uploadStatus, "error", "删除请求失败，请确认后端服务已启动");
    }
  }

  /** HTML 转义（防止 XSS） */
  function escapeHtml(str) {
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  // 刷新按钮
  btnRefreshFiles.addEventListener("click", function () {
    fetchFileList();
  });

  btnRefreshKbDocs.addEventListener("click", function () {
    fetchFileList();
  });

  // ============================================================
  // 七、笔记生成逻辑
  // ============================================================

  /** 更新字符计数 */
  queryInput.addEventListener("input", function () {
    var len = queryInput.value.length;
    queryCharCount.textContent = len + " / 500";
    if (len > 450) {
      queryCharCount.classList.add("warn");
    } else {
      queryCharCount.classList.remove("warn");
    }
  });

  btnGenerate.addEventListener("click", async function () {
    const query = queryInput.value.trim();
    const style = styleSelect.value;

    // ---- 参数校验 ----
    if (!query) {
      setStatus(generateStatus, "error", "请输入知识点或主题。");
      queryInput.focus();
      return;
    }
    if (query.length > 500) {
      setStatus(generateStatus, "error", "输入内容过长（" + query.length + "/500 字符），请精简后重试。");
      return;
    }

    // ---- 进入加载状态 ----
    setButtonLoading(btnGenerate, true);
    setStatus(generateStatus, "loading", "Agent 正在协作生成笔记...");
    // 显示 Agent 状态面板
    resetAgentStatus();
    if (agentStatusPanel) agentStatusPanel.style.display = "";
    // 隐藏上次结果
    resultSection.style.display = "none";

    var startTime = Date.now();

    try {
      // 使用 fetch + ReadableStream 接收 SSE 事件
      var response = await fetch(API_BASE + "/generate_note", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, style: style }),
      });

      if (!response.ok) {
        // 非 200 响应（校验失败等），按 JSON 处理
        var errData = await response.json();
        setStatus(generateStatus, "error", errData.msg || "请求失败");
        setButtonLoading(btnGenerate, false, "生成笔记");
        return;
      }

      // 读取 SSE 流
      var reader = response.body.getReader();
      var decoder = new TextDecoder("utf-8");
      var buffer = "";
      var finalResult = null;

      while (true) {
        var readResult = await reader.read();
        if (readResult.done) break;

        buffer += decoder.decode(readResult.value, { stream: true });

        // 按空行分割 SSE 事件块
        while (true) {
          var doubleNewlineIdx = buffer.indexOf("\n\n");
          if (doubleNewlineIdx === -1) break;

          var block = buffer.slice(0, doubleNewlineIdx);
          buffer = buffer.slice(doubleNewlineIdx + 2);

          // 解析事件块
          var eventType = "";
          var dataStr = "";
          var lines = block.split("\n");
          for (var li = 0; li < lines.length; li++) {
            var line = lines[li];
            if (line.startsWith("event: ")) {
              eventType = line.slice(7).trim();
            } else if (line.startsWith("data: ")) {
              dataStr = line.slice(6);
            }
          }

          if (!dataStr) continue;

          try {
            var eventData = JSON.parse(dataStr);

            if (eventType === "status") {
              // 实时更新 Agent 状态
              handleAgentStatus(eventData);
            } else if (eventType === "result") {
              // 最终结果
              finalResult = eventData;
            } else if (eventType === "error") {
              setStatus(generateStatus, "error", eventData.message || "工作流执行失败");
              handleAgentError(eventData);
            }
          } catch (parseErr) {
            console.warn("[SSE] 解析失败:", parseErr, dataStr);
          }
        }
      }

      // ---- SSE 流结束，处理最终结果 ----
      if (finalResult) {
        var elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        finalResult.elapsed = parseFloat(elapsed);

        if (finalResult.status === "completed") {
          setStatus(generateStatus, "success", "笔记生成成功！耗时 " + elapsed + " 秒。");
          renderResult(finalResult);
        } else if (finalResult.status === "terminated") {
          setStatus(generateStatus, "error", finalResult.error || "笔记生成终止。");
          if (finalResult.context || finalResult.framework) {
            renderResult(finalResult);
          }
        } else {
          setStatus(generateStatus, "error", finalResult.error || "生成失败，请重试。");
        }
      } else {
        setStatus(generateStatus, "error", "未收到生成结果，请重试。");
      }

    } catch (error) {
      if (error.name === "AbortError") {
        setStatus(generateStatus, "error", "生成请求超时，请重试或缩短查询内容。");
      } else {
        setStatus(generateStatus, "error", "网络请求失败，请确认后端服务已启动（http://127.0.0.1:8000）。");
      }
      console.error("[生成异常]", error);
    } finally {
      setButtonLoading(btnGenerate, false, "生成笔记");
      // 隐藏 Agent 状态面板（保留结果可见）
      if (agentStatusPanel) agentStatusPanel.style.display = "none";
    }
  });

  // ============================================================
  // 七-0、下载按钮逻辑
  // ============================================================

  btnDownload.addEventListener("click", async function () {
    if (!currentResultData) {
      resultMeta.textContent = "⚠️ 没有可下载的笔记内容。";
      return;
    }

    var noteContent = currentResultData.note || "";
    var style = currentResultData.style || "outline";
    var title = queryInput.value.trim() || "学习笔记";

    // 思维导图需要获取当前 SVG
    var svgMarkup = "";
    if (style === "mindmap" && mindmapSvg) {
      // 序列化 SVG 元素
      var serializer = new XMLSerializer();
      svgMarkup = serializer.serializeToString(mindmapSvg);
    }

    setButtonLoading(btnDownload, true, "生成文件中...");

    try {
      var response = await fetch(API_BASE + "/download_note", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          content: noteContent,
          style: style,
          svg_markup: svgMarkup,
          title: title,
        }),
      });

      if (!response.ok) {
        var errData = await response.json().catch(function () { return { msg: "下载失败" }; });
        resultMeta.textContent = "⚠️ " + (errData.msg || "文件生成失败，请重试。");
        return;
      }

      // 获取文件 blob 并触发浏览器下载
      var blob = await response.blob();
      var url = URL.createObjectURL(blob);
      var a = document.createElement("a");
      a.href = url;

      // 从 Content-Disposition 头获取服务器指定的文件名
      var disposition = response.headers.get("Content-Disposition") || "";
      var filenameMatch = disposition.match(/filename\*=UTF-8''(.+)/);
      var downloadName = "";
      if (filenameMatch) {
        downloadName = decodeURIComponent(filenameMatch[1]);
      } else {
        // 降级：自行构造文件名
        var contentType = response.headers.get("Content-Type") || "";
        var ext = contentType.indexOf("pdf") !== -1 ? ".pdf" : ".docx";
        var safeName = title.replace(/[\\/*?:"<>|]/g, "_");
        if (style === "mindmap") {
          downloadName = safeName + "思维导图" + ext;
        } else {
          var sLabel = style === "exam_points" ? "考点清单" : "提纲";
          downloadName = safeName + "-" + sLabel + ext;
        }
      }
      a.download = downloadName;

      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      resultMeta.textContent = "✅ 笔记已下载到本地。";
    } catch (error) {
      console.error("[下载异常]", error);
      resultMeta.textContent = "⚠️ 下载失败，请确认后端服务已启动。";
    } finally {
      setButtonLoading(btnDownload, false, "下载笔记");
      // 恢复按钮文字
      if (style === "mindmap") {
        btnDownloadText.textContent = "下载思维导图 (PDF)";
      } else {
        var dlLabel2 = style === "exam_points" ? "考点清单" : "提纲";
        btnDownloadText.textContent = "下载笔记 (" + dlLabel2 + " · DOCX)";
      }
    }
  });

  // ============================================================
  // 七-A、Agent 状态面板管理
  // ============================================================

  /** Agent ID → DOM 状态项映射 */
  var AGENT_STATUS_DOM = {
    "RetrievalAgent": {
      item: agentStatusRetrieval,
      text: null,
      badge: null,
    },
    "UnderstandingAgent": {
      item: agentStatusUnderstanding,
      text: null,
      badge: null,
    },
    "NoteGenerateAgent": {
      item: agentStatusGeneration,
      text: null,
      badge: null,
    },
  };

  // 初始化 DOM 引用
  for (var agentName in AGENT_STATUS_DOM) {
    var entry = AGENT_STATUS_DOM[agentName];
    if (entry.item) {
      entry.text = entry.item.querySelector(".agent-status-text");
      entry.badge = entry.item.querySelector(".agent-status-badge");
    }
  }

  /** 重置所有 Agent 状态为等待中 */
  function resetAgentStatus() {
    for (var agentName2 in AGENT_STATUS_DOM) {
      var e2 = AGENT_STATUS_DOM[agentName2];
      if (e2.item) {
        e2.item.className = "agent-status-item";
      }
      if (e2.text) {
        e2.text.textContent = "等待中...";
      }
      if (e2.badge) {
        e2.badge.textContent = "—";
        e2.badge.className = "agent-status-badge badge-pending";
      }
    }
  }

  /**
   * 处理 SSE 推送的 Agent 状态事件
   * @param {Object} event — { type: "status", agent: "RetrievalAgent", phase: "start"|"progress"|"done"|"error"|"retry", message: "..." }
   */
  function handleAgentStatus(event) {
    var agentName = event.agent;
    var phase = event.phase;
    var message = event.message;

    var entry = AGENT_STATUS_DOM[agentName];
    if (!entry || !entry.item) return;

    // 更新文本
    if (entry.text) {
      entry.text.textContent = message || "";
    }

    // 更新状态标签
    var badgeClass = "badge-pending";
    var badgeText = "—";

    if (phase === "start" || phase === "progress") {
      badgeClass = "badge-running";
      badgeText = "⚡ 进行中";
      entry.item.className = "agent-status-item is-active";
    } else if (phase === "retry") {
      badgeClass = "badge-retry";
      badgeText = "🔄 重试";
      entry.item.className = "agent-status-item is-active";
    } else if (phase === "done") {
      badgeClass = "badge-done";
      badgeText = "✅ 完成";
      entry.item.className = "agent-status-item is-done";
    } else if (phase === "error") {
      badgeClass = "badge-error";
      badgeText = "❌ 失败";
      entry.item.className = "agent-status-item is-error";
    }

    if (entry.badge) {
      entry.badge.textContent = badgeText;
      entry.badge.className = "agent-status-badge " + badgeClass;
    }
  }

  /** 处理 SSE 推送的全局错误 */
  function handleAgentError(event) {
    // 将所有未完成的 Agent 标记为错误
    for (var agentName3 in AGENT_STATUS_DOM) {
      var e3 = AGENT_STATUS_DOM[agentName3];
      if (e3.badge) {
        var badgeCls = e3.badge.className;
        // 只更新还在等待中或进行中的
        if (badgeCls.indexOf("badge-pending") !== -1 || badgeCls.indexOf("badge-running") !== -1 || badgeCls.indexOf("badge-retry") !== -1) {
          e3.badge.textContent = "❌ 中断";
          e3.badge.className = "agent-status-badge badge-error";
          e3.item.className = "agent-status-item is-error";
        }
      }
    }
  }

  // ============================================================
  // 七-B、历史笔记列表 & 查看 / 删除
  // ============================================================

  /** 当前加载的历史笔记 ID（高亮用） */
  var currentHistoryNoteId = null;

  /** 当前展示的结果数据（用于下载） */
  var currentResultData = null;

  /**
   * 获取历史笔记列表
   */
  async function fetchNotesHistory() {
    try {
      historyEmpty.textContent = "加载中...";
      historyEmpty.style.display = "";

      var response = await fetchWithTimeout(
        API_BASE + "/notes?limit=50",
        { method: "GET" },
        15000,
      );

      if (!response.ok) {
        historyEmpty.textContent = "加载失败，请刷新重试";
        return;
      }

      var data = await response.json();
      if (data.code === 200) {
        renderNotesHistory(data.notes || []);
      } else {
        historyEmpty.textContent = data.msg || "加载失败";
      }
    } catch (error) {
      console.error("[历史笔记异常]", error);
      historyEmpty.textContent = "加载失败，请确认后端服务已启动";
    }
  }

  /**
   * 渲染历史笔记列表
   * @param {Array} notes — [{ id, query, style, created_at, note_preview }]
   */
  function renderNotesHistory(notes) {
    while (historyList.firstChild) {
      historyList.removeChild(historyList.firstChild);
    }

    if (!notes || notes.length === 0) {
      historyEmpty.textContent = "暂无历史笔记，生成笔记后将自动保存";
      historyEmpty.style.display = "";
      historyList.appendChild(historyEmpty);
      return;
    }

    historyEmpty.style.display = "none";

    var styleLabels = { outline: "提纲", mindmap: "导图", exam_points: "考点" };
    var styleIcons = { outline: "📋", mindmap: "🧠", exam_points: "📝" };

    for (var i = 0; i < notes.length; i++) {
      var n = notes[i];
      var isActive = n.id === currentHistoryNoteId;

      var item = document.createElement("div");
      item.className = "history-item" + (isActive ? " active-note" : "");
      item.setAttribute("data-note-id", n.id);

      item.innerHTML =
        '<div class="history-item-icon">' + (styleIcons[n.style] || "📄") + '</div>' +
        '<div class="history-item-info">' +
          '<div class="history-item-query" title="' + escapeHtml(n.query || "") + '">' + escapeHtml(n.query || "(无标题)") + '</div>' +
          '<div class="history-item-preview">' + escapeHtml(n.note_preview || "") + '</div>' +
          '<div class="history-item-meta">' +
            '<span>' + (styleLabels[n.style] || n.style) + '</span>' +
            '<span>' + (n.created_at || "") + '</span>' +
          '</div>' +
        '</div>' +
        '<button class="history-item-delete" title="删除此笔记" data-note-id="' + escapeHtml(n.id) + '">' +
          '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">' +
            '<polyline points="2.5,4.5 13.5,4.5"/>' +
            '<path d="M5.5 4.5V3a1 1 0 011-1h3a1 1 0 011 1v1.5M11.5 4.5v8a1 1 0 01-1 1h-5a1 1 0 01-1-1v-8"/>' +
          '</svg>' +
        '</button>';

      historyList.appendChild(item);
    }

    // 绑定点击事件（查看笔记）
    var items = historyList.querySelectorAll(".history-item");
    for (var j = 0; j < items.length; j++) {
      items[j].addEventListener("click", function (e) {
        // 如果点击的是删除按钮，不触发查看
        if (e.target.closest(".history-item-delete")) return;
        var noteId = this.getAttribute("data-note-id");
        if (noteId) loadHistoryNote(noteId);
      });
    }

    // 绑定删除事件
    var delBtns = historyList.querySelectorAll(".history-item-delete");
    for (var k = 0; k < delBtns.length; k++) {
      delBtns[k].addEventListener("click", function (e) {
        e.stopPropagation();
        var noteId = this.getAttribute("data-note-id");
        if (noteId) deleteHistoryNote(noteId);
      });
    }
  }

  /**
   * 加载历史笔记完整内容到结果展示区
   * @param {string} noteId — 笔记 ID
   */
  async function loadHistoryNote(noteId) {
    try {
      var response = await fetchWithTimeout(
        API_BASE + "/notes/" + encodeURIComponent(noteId),
        { method: "GET" },
        15000,
      );

      if (!response.ok) {
        setStatus(generateStatus, "error", "加载笔记失败");
        return;
      }

      var data = await response.json();
      if (data.code === 200 && data.note) {
        // 使用现有的 renderResult 逻辑
        renderResult({
          context: data.note.context || "",
          framework: data.note.framework || "",
          note: data.note.note || "",
          style: data.note.style || "outline",
          elapsed: null,
        });

        // 高亮当前项
        currentHistoryNoteId = noteId;
        // 刷新列表以更新高亮
        fetchNotesHistory();

        // 滚动到结果区域
        resultSection.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    } catch (error) {
      console.error("[加载历史笔记异常]", error);
      setStatus(generateStatus, "error", "加载笔记失败，请确认后端服务已启动");
    }
  }

  /**
   * 删除历史笔记
   * @param {string} noteId — 笔记 ID
   */
  async function deleteHistoryNote(noteId) {
    if (!confirm("确定要删除这条历史笔记吗？此操作不可恢复。")) {
      return;
    }

    try {
      var response = await fetchWithTimeout(
        API_BASE + "/notes/" + encodeURIComponent(noteId),
        { method: "DELETE" },
        15000,
      );

      var data = await response.json();

      if (response.ok && data.code === 200) {
        // 如果删除的是当前显示的笔记，清除高亮
        if (currentHistoryNoteId === noteId) {
          currentHistoryNoteId = null;
        }
        fetchNotesHistory();
      } else {
        alert(data.msg || "删除失败");
      }
    } catch (error) {
      console.error("[删除历史笔记异常]", error);
      alert("删除失败，请确认后端服务已启动");
    }
  }

  // 刷新历史笔记按钮
  btnRefreshNotes.addEventListener("click", function () {
    fetchNotesHistory();
  });

  // ============================================================
  // 八、结果渲染
  // ============================================================

  /**
   * 渲染生成结果到页面
   * @param {Object} data - 后端返回的 JSON 数据
   * @param {string} data.context   - 原文摘要素材
   * @param {string} data.framework - 知识框架
   * @param {string} data.note      - 最终笔记
   * @param {string} data.style     - 笔记风格
   * @param {number} data.elapsed   - 耗时（秒）
   */
  function renderResult(data) {
    // ---- 显示并导航到结果区域 ----
    navigateTo("result");

    // ---- 风格标签 ----
    var styleLabels = {
      outline: "提纲",
      mindmap: "思维导图",
      exam_points: "考点清单",
    };
    resultStyleBadge.textContent = styleLabels[data.style] || data.style || "笔记";

    // ---- 填充原文素材 & 知识框架（纯文本）----
    panelContext.querySelector(".result-block").textContent =
      data.context || "（无检索到的原文素材）";

    panelFramework.querySelector(".result-block").textContent =
      data.framework || "（无知识框架）";

    // ---- 先切换到「最终笔记」标签，确保容器可见再渲染 ----
    switchTab("panelNote");

    // ---- 最终笔记：思维导图用 SVG，其他用文本 ----
    var isMindmap = data.style === "mindmap";
    var hasContent = data.note && data.note.trim();

    if (isMindmap && hasContent) {
      noteTextBlock.style.display = "none";
      mindmapContainer.style.display = "";
      renderMindmapToSvg(data.note);
    } else if (isMindmap && !hasContent) {
      // 思维导图内容为空 → 回退纯文本提示
      mindmapContainer.style.display = "none";
      noteTextBlock.style.display = "";
      noteTextBlock.textContent = data.note || "（无笔记内容 — 可能是生成超时或 token 不足，请重试）";
    } else {
      noteTextBlock.style.display = "";
      mindmapContainer.style.display = "none";
      noteTextBlock.textContent = data.note || "（无笔记内容）";
    }

    // ---- 耗时信息 ----
    if (data.elapsed) {
      resultMeta.textContent = "生成耗时：" + data.elapsed.toFixed(1) + " 秒";
    } else {
      resultMeta.textContent = "";
    }

    // ---- 滚动到结果区域 ----
    resultSection.scrollIntoView({ behavior: "smooth", block: "start" });

    // ---- 保存结果数据供下载使用 ----
    currentResultData = data;

    // ---- 显示下载按钮 ----
    if (btnDownload && data.note && data.note.trim()) {
      btnDownload.style.display = "inline-flex";
      if (data.style === "mindmap") {
        btnDownloadText.textContent = "下载思维导图 (PDF)";
      } else {
        var dlLabel = data.style === "exam_points" ? "考点清单" : "提纲";
        btnDownloadText.textContent = "下载笔记 (" + dlLabel + " · DOCX)";
      }
    }

    // ---- 刷新历史笔记列表 ----
    fetchNotesHistory();
  }

  /**
   * 纯 JS SVG 思维导图渲染器（零外部依赖）。
   * 将 Markdown # 层级标题解析为水平树形布局，绘制节点 + 贝塞尔连线。
   */
  function renderMindmapToSvg(markdown) {
    console.log("[Mindmap] 开始渲染，原文长度=" + markdown.length);
    console.log("[Mindmap] 原文预览=" + markdown.slice(0, 200));

    try {
      // ---- 0. 清除旧内容（用 DOM 方式，不用 innerHTML）----
      while (mindmapSvg.firstChild) {
        mindmapSvg.removeChild(mindmapSvg.firstChild);
      }

      // ---- 1. 解析 Markdown → 树（兼容新旧两种格式）----
      var lines = markdown.split("\n");
      var root = { text: "", depth: 0, children: [], parent: null };
      var stack = [root];

      for (var i = 0; i < lines.length; i++) {
        var raw = lines[i];
        var line = raw.trim();
        if (!line) continue;

        var depth, text, match;

        // 格式A：Markdown 标题 # / ## / ### / ####
        match = line.match(/^(#{1,4})\s*(.+)/);
        if (match) {
          depth = match[1].length;
          text = match[2].trim();
        } else {
          // 格式B：缩进列表 ★ / - /   - /     -（旧模板兼容）
          match = raw.match(/^(\s*)(?:[-▶▷·★→⇨\*])\s*(.+)/);
          if (match) {
            var leading = match[1];
            if (line.charAt(0) === "★") {
              depth = 1;
            } else {
              depth = Math.floor(leading.length / 2) + 2;
            }
            text = match[2].trim();
          } else {
            continue;
          }
        }

        // 清理残留符号
        text = text.replace(/^[★▶▷·→⇨\-\*]\s*/, "");
        if (!text) continue;
        if (text.length > 22) text = text.slice(0, 20) + "…";

        var node = { text: text, depth: depth, children: [], parent: null };

        // 找到父节点
        while (stack.length > depth) stack.pop();
        if (stack.length === 0) stack.push(root);
        var parent = stack[stack.length - 1];
        node.parent = parent;
        parent.children.push(node);
        stack.push(node);
      }

      console.log("[Mindmap] 解析完成，根节点子节点数=" + root.children.length);

      // 无有效节点 → 回退纯文本（带原文诊断）
      if (root.children.length === 0) {
        console.warn("[Mindmap] 未解析到节点，原文前300字=" + markdown.slice(0, 300));
        noteTextBlock.style.display = "";
        mindmapContainer.style.display = "none";
        noteTextBlock.textContent = markdown;
        return;
      }

      // ---- 2. 布局计算 ----
      var levelGapX = 200;
      var nodeGapY = 14;
      var nodeH = 34;
      var nodeRx = 8;

      function layout(node, x, topY) {
        if (node.children.length === 0) {
          node._x = x;
          node._y = topY + nodeH / 2;
          node._w = 0;
          return topY + nodeH + nodeGapY;
        }
        var childTop = topY;
        for (var ci = 0; ci < node.children.length; ci++) {
          childTop = layout(node.children[ci], x + levelGapX, childTop);
        }
        var first = node.children[0];
        var last = node.children[node.children.length - 1];
        node._x = x;
        node._y = (first._y + last._y) / 2;
        return childTop;
      }

      var centerNode = root.children[0];
      layout(centerNode, 30, 20);

      // ---- 3. 测量文字宽度 ----
      function textWidth(text) {
        var w = 0;
        for (var ti = 0; ti < text.length; ti++) {
          var c = text.charCodeAt(ti);
          w += (c >= 0x4e00 && c <= 0x9fff) || c >= 0x2000 ? 14 : 8;
        }
        return w + 24;
      }

      // ---- 4. 收集节点 & 计算尺寸 ----
      var allNodes = [];
      var maxX = 0, maxY = 0;
      (function collect(n) {
        n._w = textWidth(n.text);
        if (n._x + n._w > maxX) maxX = n._x + n._w;
        if (n._y > maxY) maxY = n._y;
        allNodes.push(n);
        for (var ci2 = 0; ci2 < n.children.length; ci2++) collect(n.children[ci2]);
      })(centerNode);

      var svgW = Math.max(maxX + 60, 400);
      var svgH = Math.max(maxY + nodeH + 20, 100);

      console.log("[Mindmap] 布局完成 | 节点数=" + allNodes.length + " | 尺寸=" + svgW + "x" + svgH);

      // ---- 5. 渲染 ----
      var SVG_NS = "http://www.w3.org/2000/svg";

      mindmapSvg.setAttribute("viewBox", "0 0 " + svgW + " " + svgH);
      mindmapSvg.setAttribute("width", "100%");
      mindmapSvg.setAttribute("height", svgH);
      mindmapSvg.style.display = "block";

      // defs
      var defs = document.createElementNS(SVG_NS, "defs");

      // 阴影
      var filter = document.createElementNS(SVG_NS, "filter");
      filter.setAttribute("id", "nodeShadow");
      filter.setAttribute("x", "-10%"); filter.setAttribute("y", "-10%");
      filter.setAttribute("width", "130%"); filter.setAttribute("height", "130%");
      var feDrop = document.createElementNS(SVG_NS, "feDropShadow");
      feDrop.setAttribute("dx", "1"); feDrop.setAttribute("dy", "2");
      feDrop.setAttribute("stdDeviation", "2");
      feDrop.setAttribute("flood-opacity", "0.15");
      filter.appendChild(feDrop);
      defs.appendChild(filter);

      mindmapSvg.appendChild(defs);

      // 背景
      var bg = document.createElementNS(SVG_NS, "rect");
      bg.setAttribute("x", "0"); bg.setAttribute("y", "0");
      bg.setAttribute("width", "100%"); bg.setAttribute("height", "100%");
      bg.setAttribute("fill", "#f8fafc");
      mindmapSvg.appendChild(bg);

      // 连线
      var linesG = document.createElementNS(SVG_NS, "g");
      for (var li = 0; li < allNodes.length; li++) {
        var nd = allNodes[li];
        if (!nd.parent || nd.parent === root) continue;
        var px = nd.parent._x + nd.parent._w;
        var py = nd.parent._y;
        var cx = nd._x;
        var cy = nd._y;
        var mx = (px + cx) / 2;
        var path = document.createElementNS(SVG_NS, "path");
        path.setAttribute("d",
          "M" + px + "," + py +
          " C" + mx + "," + py + " " + mx + "," + cy + " " + cx + "," + cy);
        path.setAttribute("stroke", "#cbd5e1");
        path.setAttribute("stroke-width", "2");
        path.setAttribute("fill", "none");
        linesG.appendChild(path);
      }
      mindmapSvg.appendChild(linesG);

      // 节点
      var colors = [
        { fill: "#4f6ef7", text: "#fff" },
        { fill: "#f59e0b", text: "#fff" },
        { fill: "#10b981", text: "#fff" },
        { fill: "#8b5cf6", text: "#fff" },
        { fill: "#ec4899", text: "#fff" },
      ];
      var nodesG = document.createElementNS(SVG_NS, "g");
      for (var ni = 0; ni < allNodes.length; ni++) {
        var nd2 = allNodes[ni];
        var col = colors[Math.min(nd2.depth - 1, colors.length - 1)];

        var rect = document.createElementNS(SVG_NS, "rect");
        rect.setAttribute("x", nd2._x);
        rect.setAttribute("y", nd2._y - nodeH / 2);
        rect.setAttribute("width", nd2._w);
        rect.setAttribute("height", nodeH);
        rect.setAttribute("rx", nodeRx);
        rect.setAttribute("fill", col.fill);
        rect.setAttribute("filter", "url(#nodeShadow)");
        rect.style.cursor = "pointer";
        nodesG.appendChild(rect);

        var txt = document.createElementNS(SVG_NS, "text");
        txt.setAttribute("x", nd2._x + nd2._w / 2);
        txt.setAttribute("y", nd2._y + 5);
        txt.setAttribute("text-anchor", "middle");
        txt.setAttribute("fill", col.text);
        txt.setAttribute("font-size", "13");
        txt.setAttribute("font-family", "PingFang SC, Microsoft YaHei, sans-serif");
        txt.setAttribute("font-weight", nd2.depth <= 2 ? "600" : "400");
        txt.textContent = nd2.text;
        nodesG.appendChild(txt);
      }
      mindmapSvg.appendChild(nodesG);

      console.log("[Mindmap] ✅ SVG 渲染成功 | 节点=" + allNodes.length + " | " + svgW + "x" + svgH);

      // 新导图渲染后重置缩放
      resetMindmapZoom();

    } catch (err) {
      console.error("[Mindmap] ❌ 渲染异常:", err);
      // 出错时回退纯文本
      noteTextBlock.style.display = "";
      mindmapContainer.style.display = "none";
      noteTextBlock.textContent = markdown;
    }
  }

  // ============================================================
  // 八-B、思维导图缩放与拖拽控制
  // ============================================================

  /** 应用当前缩放比 + 平移量到 SVG 包装层 */
  function applyMindmapTransform() {
    if (!mindmapSvgWrapper) return;
    mindmapSvgWrapper.style.transform =
      "translate(" + mmPanX + "px, " + mmPanY + "px) scale(" + mmZoom + ")";
    if (zoomLevelEl) {
      zoomLevelEl.textContent = Math.round(mmZoom * 100) + "%";
    }
    // 同步包装层布局尺寸，确保容器滚动条匹配缩放后视觉大小
    if (mindmapSvg && mindmapSvg.viewBox && mindmapSvg.viewBox.baseVal) {
      var vb = mindmapSvg.viewBox.baseVal;
      if (vb.width > 0 && vb.height > 0) {
        mindmapSvgWrapper.style.width  = (vb.width  * mmZoom) + "px";
        mindmapSvgWrapper.style.height = (vb.height * mmZoom) + "px";
      }
    }
  }

  /** 缩放至指定级别 */
  function setMindmapZoom(level) {
    mmZoom = Math.min(MM_ZOOM_MAX, Math.max(MM_ZOOM_MIN, level));
    applyMindmapTransform();
  }

  /** 放大 */
  function zoomMindmapIn() {
    setMindmapZoom(mmZoom + MM_ZOOM_STEP);
  }

  /** 缩小 */
  function zoomMindmapOut() {
    setMindmapZoom(mmZoom - MM_ZOOM_STEP);
  }

  /** 重置缩放与平移 */
  function resetMindmapZoom() {
    mmZoom = 1.0;
    mmPanX = 0;
    mmPanY = 0;
    applyMindmapTransform();
  }

  // ---- 缩放按钮事件 ----
  if (btnZoomIn) {
    btnZoomIn.addEventListener("click", function (e) {
      e.stopPropagation();
      zoomMindmapIn();
    });
  }
  if (btnZoomOut) {
    btnZoomOut.addEventListener("click", function (e) {
      e.stopPropagation();
      zoomMindmapOut();
    });
  }
  if (btnZoomReset) {
    btnZoomReset.addEventListener("click", function (e) {
      e.stopPropagation();
      resetMindmapZoom();
    });
  }

  // ---- 鼠标拖拽平移 ----
  if (mindmapContainer) {
    mindmapContainer.addEventListener("mousedown", function (e) {
      // 只在思维导图可见时响应拖拽；忽略按钮点击
      if (mindmapContainer.style.display === "none") return;
      if (e.target.closest(".mindmap-zoom-controls")) return;
      e.preventDefault();
      mmDragging = true;
      mmDragStartX = e.clientX;
      mmDragStartY = e.clientY;
      mmDragStartPanX = mmPanX;
      mmDragStartPanY = mmPanY;
      mindmapContainer.classList.add("is-dragging");
    });

    window.addEventListener("mousemove", function (e) {
      if (!mmDragging) return;
      var dx = e.clientX - mmDragStartX;
      var dy = e.clientY - mmDragStartY;
      mmPanX = mmDragStartPanX + dx;
      mmPanY = mmDragStartPanY + dy;
      applyMindmapTransform();
    });

    window.addEventListener("mouseup", function () {
      if (mmDragging) {
        mmDragging = false;
        mindmapContainer.classList.remove("is-dragging");
      }
    });

    // 鼠标滚轮缩放
    mindmapContainer.addEventListener("wheel", function (e) {
      if (mindmapContainer.style.display === "none") return;
      if (Math.abs(e.deltaY) < Math.abs(e.deltaX)) return;
      e.preventDefault();
      if (e.deltaY < 0) {
        zoomMindmapIn();
      } else {
        zoomMindmapOut();
      }
    }, { passive: false });
  }

  // ============================================================
  // 九、标签切换逻辑
  // ============================================================

  resultTabs.addEventListener("click", function (e) {
    const tab = e.target.closest(".tab");
    if (!tab) return;

    const targetId = tab.getAttribute("data-tab");
    if (targetId) {
      switchTab(targetId);
    }
  });

  /**
   * 切换到指定标签页
   * @param {string} tabId - 目标内容面板的 ID
   */
  function switchTab(tabId) {
    // 更新标签激活状态
    var tabs = resultTabs.querySelectorAll(".tab");
    tabs.forEach(function (t) {
      if (t.getAttribute("data-tab") === tabId) {
        t.classList.add("active");
      } else {
        t.classList.remove("active");
      }
    });

    // 更新内容面板
    var panels = resultSection.querySelectorAll(".tab-content");
    panels.forEach(function (p) {
      if (p.id === tabId) {
        p.classList.add("active");
      } else {
        p.classList.remove("active");
      }
    });
  }

  // ============================================================
  // 十、键盘快捷键（可选增强）
  // ============================================================

  // Ctrl+Enter 触发生成按钮
  queryInput.addEventListener("keydown", function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      if (!btnGenerate.disabled) {
        btnGenerate.click();
      }
    }
  });

  // ============================================================
  // 十一、页面初始化
  // ============================================================

  // 初始显示主页（隐藏上传区和结果区）
  navigateTo("home");

  // 后台预加载文档列表和历史笔记
  fetchFileList();
  fetchNotesHistory();

  console.log("本地知识库学习笔记生成助手 — 前端已就绪");
  console.log("API 地址：" + API_BASE);
  console.log("支持格式：" + ALLOWED_EXTENSIONS.join("、"));
})();
