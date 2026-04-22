# 公开发布说明

这份文档针对“把 NASAI 作为公开仓库发布”，不是针对你的真实图库运行结果发布。

## 公开仓库里应保留什么

- CLI 源码
- 本地预览源码
- `benchmark` 实验脚本
- HTML 模板
- 配置示例
- 文档
- 依赖声明

## 公开仓库里绝不能出现什么

- `.env`
- Immich API Key
- 本地数据库 `nasai.db*`
- `cache/` 下的缩略图缓存
- `preview/` 下的本地编译产物
- `benchmark/results/` 下的所有报告和导出目录
- `logs/` 和 `backups/`
- 任何真实文件名、真实原始路径、真实人物名、真实缩略图

## 推送前检查

1. 确认 `.gitignore` 已覆盖数据库、缓存、预览结果和实验输出。
2. 把 `.env.example` 里的真实主机和本地路径替换成占位值。
3. 检查 `README`、`benchmark` 脚本和 `launchd` 模板里没有写入你的私有 NAS 地址、用户名或本机绝对路径。
4. 用 `git status --short` 看一遍待提交内容。
5. 再创建公开远端并推送。

## LaunchAgent 模板

公开仓库可以保留 `launchd/com.example.nasai.incremental.plist` 这种模板文件，但不要提交你实际安装到 `~/Library/LaunchAgents/` 的那份用户级 plist。

## 推荐发布顺序

```bash
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin <your-public-repo-url>
git push -u origin main
```

## 许可证

如果你希望别人可以 fork、复用或二次分发，公开推送前应该补一个明确许可证。

如果只是“公开可见但暂不授权复用”，也可以先不加许可证，但这会限制他人的合法使用范围。
