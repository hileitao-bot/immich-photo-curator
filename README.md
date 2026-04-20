# NASAI

NASAI 是一套运行在 macOS 本地的 Immich 辅助工具，目标是把“大图库筛选”拆成三层：

- 从 Immich 拉取元数据到本地 SQLite
- 在 Mac 上对缩略图做本地评分、中文标签和中文搜索索引
- 先在本地 HTML 预览里做分层，再决定是否把结果写回 Immich

这个项目的默认立场是保守：

- 不写源媒体文件
- 先本地预览，再写回 Immich
- 外部库是否只读，不影响本地评分和预览流程

## 当前能力

- `discover`: 拉取 Immich 资产元数据到本地索引
- `score`: 下载缩略图，在本地做 Vision 分析、中文标签和分数归一化
- `preview`: 启动本地预览页
- `dedupe`: 对连拍/近重复图做本地去重
- `sync-trial`: 把试跑结果同步到 Immich 试点相册
- `sync-tags`: 把等级和中文标签写回 Immich 标签
- `apply-archive`: 按阈值把资产切到 `timeline` / `archive`
- `benchmark/run_hybrid_trial.py`: 生成混合筛选报告、过滤页和系统缓冲导出目录

## 安全边界

- NASAI 不会修改 NAS 上的原始照片或视频文件。
- 本地评分基于缩略图缓存和本地 SQLite。
- 会写回的对象只有 Immich 元数据层，且仅在你显式执行 `sync-*` 或 `apply-archive` 时发生。
- 公开仓库不应包含真实数据库、缩略图缓存、结果页、系统缓冲导出物或 API 密钥。

更完整的说明见：

- [混合筛选流程](./docs/hybrid-workflow.md)
- [公开发布说明](./docs/public-release.md)

## 环境要求

- macOS
- Python 3.9+
- `uv`
- Xcode Command Line Tools 或可用的 `swiftc`

主 CLI 依赖安装：

```bash
uv sync
```

如果要跑 `benchmark` 目录下的实验脚本，需要 Python 3.10+ 和额外的实验依赖：

```bash
uv sync --extra benchmark
```

## 配置

复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

然后填写：

- `IMMICH_BASE_URL`
- `IMMICH_API_KEY`
- `NASAI_DB_PATH`
- `NASAI_CACHE_DIR`
- `NASAI_PREVIEW_DIR`

## 快速开始

1. 拉取一批元数据到本地：

```bash
uv run nasai discover --limit 1000
```

2. 在本地下载缩略图并评分：

```bash
uv run nasai score --limit 1000
```

3. 对图片做一次连拍去重：

```bash
uv run nasai dedupe
```

4. 启动本地预览：

```bash
uv run nasai preview
```

5. 只在确认后，再把试点结果同步回 Immich：

```bash
uv run nasai sync-trial
uv run nasai sync-tags --limit 300
uv run nasai apply-archive --threshold 0.8
```

## Hybrid 预览

`benchmark/run_hybrid_trial.py` 用于更激进的本地预览实验。它会把资产分成：

- 精选展示
- 系统缓冲
- 低优先级
- 归档样本
- 视频保护
- 精选视频
- 系统缓冲视频

同时会把系统缓冲图片和视频额外导出到单独目录，便于二次筛选：

- `benchmark/results/hybrid/system_buffer/images/`
- `benchmark/results/hybrid/system_buffer/videos/`
- `benchmark/results/hybrid/system_buffer/manifest.json`

详情见 [混合筛选流程](./docs/hybrid-workflow.md)。

## 目录结构

```text
nasai/
├── nasai/                  # 主 CLI 和本地预览应用
├── benchmark/              # 实验脚本与 HTML 模板
├── tools/vision_probe.swift
├── .env.example
├── pyproject.toml
└── docs/
```

## 公开发布注意事项

- 不要提交 `.env`
- 不要提交 `nasai.db*`
- 不要提交 `cache/`、`preview/`
- 不要提交 `benchmark/results/`
- 不要提交任何包含真实文件名、真实路径或真实缩略图的产物

如果你希望别人可以自由复用这套代码，公开推送前还应补一个明确许可证。
