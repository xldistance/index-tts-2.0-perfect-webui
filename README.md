<img width="3786" height="1748" alt="QQ20250911-151718" src="https://github.com/user-attachments/assets/bfc8a686-cb14-4ee9-bb7d-27c87e965ac1" />
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
   * 可以按队列顺序生成和播放音频。

3. **历史生成音频管理：**

   * 保留**最近6个生成的音频**文件。
   * **自动刷新**的情况下，如果有新的音频生成，历史音频播放将被打断。

---
