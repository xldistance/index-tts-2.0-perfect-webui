
<img width="3714" height="3064" alt="FireShot Capture 025 - IndexTTS 2 0 -  127 0 0 1" src="https://github.com/user-attachments/assets/dfb08c1b-fb90-4137-86f3-c496cf78bedb" />
# 推荐安装deepspeed加速
https://github.com/6Morpheus6/deepspeed-windows-wheels

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
