<img width="3754" height="1786" alt="QQ20260113-172527" src="https://github.com/user-attachments/assets/d757fd63-e1e5-4fe4-8482-a1df44eb61a6" />

# 📌 升级建议：Gradio 与 Pandas

为确保项目稳定运行并兼容最新功能，**强烈建议**将以下两个关键库升级至最新版本：

## ✅ 升级 Gradio

```bash
pip install --upgrade gradio
```

> Gradio 最新版本提供更流畅的 UI 交互、性能优化及新组件支持。

---

## ⚠️ Pandas 升级（重要！）

若在运行过程中 **出现 Pandas 相关报错**，请**务必**将其升级至最新版本：

```bash
pip install --upgrade pandas
```
核心新增功能包括：

1. **从音色参考音频文件目录导入音频：**

   * 音色参考音频文件目录为 `saved_timbres`。
   * 系统将从该目录中导入音色参考音频。

2. **队列生成音频：**

   * 支持音频生成队列的管理。
