# Hybrid 混合筛选流程

这套流程是为“大体量 Immich 资产先本地预览、后决定是否写回”设计的。

## 目标

- 不碰 NAS 源文件
- 尽量只依赖本地缓存和 Immich 元数据
- 在大图库里先做“展示层级”而不是直接删

## 输入

- `nasai.db` 中的资产元数据
- 本地缩略图缓存
- Immich 已有的人脸识别、命名人物和中文标签线索
- 本地 Vision 分析结果

## 主输出

运行：

```bash
uv run python benchmark/run_hybrid_trial.py
```

会生成：

- `benchmark/results/hybrid/index.html`
- `benchmark/results/hybrid/filtered.html`
- `benchmark/results/hybrid/actions.json`
- `benchmark/results/hybrid/system_buffer/`

这些输出都是本地产物，不应该进入公开仓库。

## 图片分层

图片会先经过：

1. 文档/截图类下沉
2. 连拍与近重复归并
3. 结合命名人物、人脸质量、美学分和原始分数排序
4. 把展示层限制在更少的代表图

当前主要层级：

- `精选展示`: 真正进入主展示页
- `系统缓冲`: 没进精选，但也不直接归档
- `低优先级`: 质量较低但仍保留在过滤页
- `归档样本`: 文档、截图、票据、无意义图等

## 视频分层

视频会优先保护：

- 命名人物视频
- 命中 `合照/儿童/宝宝` 且检测到人脸的视频
- vlog / 旅行命名视频
- 长视频或明显有叙事感的视频

其余视频再分成：

- `精选视频`
- `系统缓冲视频`

## 系统缓冲导出

系统缓冲层会被额外导出到：

- `benchmark/results/hybrid/system_buffer/images/`
- `benchmark/results/hybrid/system_buffer/videos/`

目录里保存的是缩略图或视频封面，不是原始媒体。

同目录下还会生成：

- `manifest.json`: 包含 `assetId`、文件名、原路径、人物、标签、打开 Immich 的链接和当前判定原因
- `README.txt`: 便于后续人工二次筛选

## 为什么要有系统缓冲

30 万规模下，不现实要求人工逐条复核。系统缓冲层的目的不是“等人精审”，而是：

- 避免把所有未入选内容都直接打成归档
- 给后续更保守或更激进的规则留回旋空间
- 在需要时导出一批值得二次筛的边缘内容

## 写回 Immich 的边界

Hybrid 报告本身只生成本地结果。

是否把结果写回 Immich，应该拆成单独步骤：

- 先数据库备份
- 再按你确认过的规则同步标签、相册或可见性
- 始终不写源文件
