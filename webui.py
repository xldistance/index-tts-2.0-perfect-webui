import json
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
import queue
import uuid
from pathlib import Path
import warnings
import re
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7818, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--use_deepspeed", action="store_true", default=False, help="Use Deepspeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use cuda kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=200, help="GUI: Max tokens per generation segment")
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

for file in [
    "bpe.model",
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="Auto")
MODE = 'local'
tts = IndexTTS2(model_dir=cmd_args.model_dir,
                cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
                use_fp16=cmd_args.fp16,
                use_deepspeed=cmd_args.use_deepspeed,
                use_cuda_kernel=cmd_args.cuda_kernel,
                )

# 支持的语言列表
LANGUAGES = {
    "中文": "zh_CN",
    "English": "en_US"
}
EMO_CHOICES = [i18n("与音色参考音频相同"),
               i18n("使用情感参考音频"),
               i18n("使用情感向量控制"),
               i18n("使用情感描述文本控制")]

os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("prompts", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# 新增：从 saved_timbres 目录加载音色文件
SAVED_TIMBRES_DIR = os.path.join(current_dir, "saved_timbres")
os.makedirs(SAVED_TIMBRES_DIR, exist_ok=True)

SUPPORTED_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

# 生成历史记录管理
generation_history = deque(maxlen=10)  # 增加历史记录数量
generation_lock = threading.Lock()

# ========== 队列系统相关变量 ==========
task_queue = queue.Queue()
queue_status = {}
queue_lock = threading.Lock()
processing_thread = None
stop_processing = False
current_task_id = None

# 任务状态枚举
class TaskStatus:
    PENDING = "等待中"
    PROCESSING = "生成中"
    COMPLETED = "已完成"
    FAILED = "失败"
    CANCELLED = "已取消"

def list_timbres():
    """返回 saved_timbres 目录下的音频文件绝对路径列表（按文件名排序）。"""
    files = []
    if os.path.isdir(SAVED_TIMBRES_DIR):
        for fn in os.listdir(SAVED_TIMBRES_DIR):
            if fn.lower().endswith(SUPPORTED_AUDIO_EXTS):
                files.append(os.path.join(SAVED_TIMBRES_DIR, fn))
    files.sort(key=lambda p: os.path.basename(p).lower())
    return files

def get_default_timbre():
    """获取默认音色文件，优先返回'甜美女声1.mp3'"""
    timbres = list_timbres()
    sweet_voice_path = os.path.join(SAVED_TIMBRES_DIR, "甜美女声1.mp3")
    
    if sweet_voice_path in timbres:
        return sweet_voice_path
    return timbres[0] if timbres else None

# 预计算下拉默认项
timbre_choices_boot = list_timbres()
default_timbre_boot = get_default_timbre()

MAX_LENGTH_TO_USE_SPEED = 70

# 读取示例
with open("examples/cases.jsonl", "r", encoding="utf-8") as f:
    example_cases = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        if example.get("emo_audio", None):
            emo_audio_path = os.path.join("examples", example["emo_audio"])
        else:
            emo_audio_path = None
        example_cases.append([
            default_timbre_boot,
            EMO_CHOICES[example.get("emo_mode", 0)],
            example.get("text"),
            emo_audio_path,
            example.get("emo_weight", 1.0),
            example.get("emo_text", ""),
            example.get("emo_vec_1", 0),
            example.get("emo_vec_2", 0),
            example.get("emo_vec_3", 0),
            example.get("emo_vec_4", 0),
            example.get("emo_vec_5", 0),
            example.get("emo_vec_6", 0),
            example.get("emo_vec_7", 0),
            example.get("emo_vec_8", 0)
        ])

def add_to_history(audio_path):
    """添加新生成的音频到历史记录"""
    with generation_lock:
        generation_history.append({
            'path': audio_path,
            'time': datetime.now(),
            'text': ''  # 可以存储生成的文本
        })

def continuous_queue_refresh():
    """持续刷新队列状态的生成器函数"""
    while True:
        time.sleep(2)
        yield get_queue_status()

def get_history_display():
    """获取历史记录的显示格式"""
    with generation_lock:
        if not generation_history:
            return [None] * 6
        
        history_list = list(generation_history)
        history_list.reverse()
        
        result = [None] * 6
        for i, item in enumerate(history_list[:6]):
            if i < 6:
                result[i] = item['path']
        
        return result

def refresh_history():
    """刷新历史记录显示"""
    history = get_history_display()
    return [gr.update(value=h, visible=h is not None) for h in history]

# ========== 队列处理相关函数 ==========
def process_queue():
    """后台处理队列中的任务"""
    global current_task_id, stop_processing
    
    while not stop_processing:
        try:
            task = task_queue.get(timeout=1)
            
            if task is None:
                break
                
            task_id = task['id']
            current_task_id = task_id
            
            with queue_lock:
                if task_id in queue_status:
                    if queue_status[task_id]['status'] == TaskStatus.CANCELLED:
                        continue
                    queue_status[task_id]['status'] = TaskStatus.PROCESSING
                    queue_status[task_id]['start_time'] = datetime.now()
            
            try:
                output = gen_single_core(task['params'])
                
                with queue_lock:
                    if task_id in queue_status:
                        queue_status[task_id]['status'] = TaskStatus.COMPLETED
                        queue_status[task_id]['output'] = output
                        queue_status[task_id]['end_time'] = datetime.now()
                        
                if output:
                    add_to_history(output)
                    
            except Exception as ex:
                print(f"队列生成音频失败，错误信息：{ex}")
                with queue_lock:
                    if task_id in queue_status:
                        queue_status[task_id]['status'] = TaskStatus.FAILED
                        queue_status[task_id]['error'] = str(ex)
                        queue_status[task_id]['end_time'] = datetime.now()
            
            current_task_id = None
            
        except queue.Empty:
            continue
        except Exception as ex:
            print(f"Queue processing error: {ex}")
            current_task_id = None

def gen_single_core(params):
    """核心生成函数"""
    emo_control_method = params['emo_control_method']
    prompt = params['prompt']
    text = params['text']
    emo_ref_path = params['emo_ref_path']
    emo_weight = params['emo_weight']
    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8 = params['vec']
    emo_text = params['emo_text']
    emo_random = params['emo_random']
    max_text_tokens_per_segment = params['max_text_tokens_per_segment']
    kwargs = params['kwargs']
    
    timestamp = int(time.time() * 1000)
    cleaned_text = re.sub(r'[\n ]', '', text)
    output_path = os.path.join("outputs", f"{Path(prompt).stem}_{cleaned_text[:20]}_{timestamp}.wav")
    
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:
        emo_ref_path = None
        emo_weight = 1.0
    if emo_control_method == 1:
        emo_weight = emo_weight
    if emo_control_method == 2:
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        vec_sum = sum([vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8])
        if vec_sum > 1.5:
            raise ValueError(i18n("情感向量之和不能超过1.5，请调整后重试。"))
    else:
        vec = None

    if emo_text == "":
        emo_text = None

    print(f"Emo control mode:{emo_control_method}, vec:{vec}")
    
    output = tts.infer(spk_audio_prompt=prompt, text=text,
                      output_path=output_path,
                      emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight,
                      emo_vector=vec,
                      use_emo_text=(emo_control_method == 3), emo_text=emo_text, use_random=emo_random,
                      verbose=cmd_args.verbose,
                      max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                      **kwargs)
    
    return output

def add_to_queue(emo_control_method, prompt, text,
                 emo_ref_path, emo_weight,
                 vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                 emo_text, emo_random,
                 max_text_tokens_per_segment,
                 *args):
    """添加任务到队列"""
    global processing_thread, stop_processing
    
    do_sample, top_p, top_k, temperature, \
        length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": num_beams,
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }
    
    task_id = str(uuid.uuid4())
    task = {
        'id': task_id,
        'params': {
            'emo_control_method': emo_control_method,
            'prompt': prompt,
            'text': text,
            'emo_ref_path': emo_ref_path,
            'emo_weight': emo_weight,
            'vec': [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8],
            'emo_text': emo_text,
            'emo_random': emo_random,
            'max_text_tokens_per_segment': max_text_tokens_per_segment,
            'kwargs': kwargs
        }
    }
    
    with queue_lock:
        queue_status[task_id] = {
            'status': TaskStatus.PENDING,
            'text': text[:50] + '...' if len(text) > 50 else text,
            'submit_time': datetime.now(),
            'position': task_queue.qsize() + 1
        }
    
    task_queue.put(task)
    
    if processing_thread is None or not processing_thread.is_alive():
        stop_processing = False
        processing_thread = threading.Thread(target=process_queue, daemon=True)
        processing_thread.start()
    
    return get_queue_status()

def get_queue_status():
    """获取队列状态信息"""
    with queue_lock:
        pending_count = sum(1 for s in queue_status.values() if s['status'] == TaskStatus.PENDING)
        processing_count = sum(1 for s in queue_status.values() if s['status'] == TaskStatus.PROCESSING)
        completed_count = sum(1 for s in queue_status.values() if s['status'] == TaskStatus.COMPLETED)

        data = []
        sorted_tasks = sorted(queue_status.items(), key=lambda x: x[1]['submit_time'])

        for idx, (task_id, info) in enumerate(sorted_tasks[-10:], 1):
            status_emoji = {
                TaskStatus.PENDING: "⏳",
                TaskStatus.PROCESSING: "🔄",
                TaskStatus.COMPLETED: "✅",
                TaskStatus.FAILED: "❌",
                TaskStatus.CANCELLED: "🚫"
            }.get(info['status'], "")

            data.append([
                idx,
                info['text'],
                f"{status_emoji} {info['status']}",
                info['submit_time'].strftime("%H:%M:%S")
            ])

        status_text = f"""
        <div style='padding: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
            <h4 style='margin: 0 0 10px 0;'>📊 队列状态</h4>
            <div style='display: flex; justify-content: space-around;'>
                <div>🔄 处理中: <b>{processing_count}</b></div>
                <div>⏳ 等待中: <b>{pending_count}</b></div>
                <div>✅ 已完成: <b>{completed_count}</b></div>
                <div>📁 队列长度: <b>{task_queue.qsize()}</b></div>
            </div>
        </div>
        """

        latest_output = None
        for task_id, info in reversed(sorted_tasks):
            if info['status'] == TaskStatus.COMPLETED and 'output' in info:
                latest_output = info['output']
                break

        hist1, hist2, hist3, hist4, hist5, hist6 = get_history_display()

        queue_update = gr.update(value=status_text)
        table_update = gr.update(value=data)
        latest_output_update = gr.update(value=latest_output, visible=bool(latest_output)) if latest_output else gr.update()
        
        history_updates = []
        for hist in [hist1, hist2, hist3, hist4, hist5, hist6]:
            history_updates.append(gr.update(value=hist, visible=bool(hist)))

        return (queue_update, table_update, latest_output_update, *history_updates)

def clear_completed_tasks():
    """清除已完成的任务"""
    with queue_lock:
        to_remove = [tid for tid, info in queue_status.items() 
                    if info['status'] in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]]
        for tid in to_remove:
            del queue_status[tid]
    return get_queue_status()

def on_input_text_change(text, max_text_tokens_per_segment):
    if text and len(text) > 0:
        text_tokens_list = tts.tokenizer.tokenize(text)
        segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
        data = []
        for i, s in enumerate(segments):
            segment_str = ''.join(s)
            tokens_count = len(s)
            data.append([i, segment_str, tokens_count])
        return gr.update(value=data, visible=True)
    else:
        return gr.update(value=[])

def on_method_select(emo_control_method):
    if emo_control_method == 1:
        return (gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False))
    elif emo_control_method == 2:
        return (gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False))
    elif emo_control_method == 3:
        return (gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True))
    else:
        return (gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False))

def update_timbre_preview(selected_path):
    """根据下拉选中的路径，更新试听播放器"""
    if selected_path and os.path.exists(selected_path):
        return gr.update(value=selected_path, visible=True)
    return gr.update(value=None, visible=False)

def refresh_timbres():
    """刷新 saved_timbres 列表"""
    choices = list_timbres()
    value = get_default_timbre()
    dropdown_update = gr.update(choices=choices, value=value)
    preview_update = gr.update(value=value, visible=bool(value))
    return dropdown_update, preview_update

# 自定义CSS样式
custom_css = """
    /* 渐变背景 */
    .gradio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* 主容器样式 */
    .container {
        max-width: 1400px !important;
    }
    
    /* 标签页样式 */
    .tabs {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    /* 按钮样式 */
    .primary-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: bold !important;
        transition: transform 0.2s !important;
    }
    
    .primary-btn:hover {
        transform: scale(1.05) !important;
    }
    
    /* 卡片样式 */
    .card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* 表格样式 */
    .dataframe {
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    
    /* 音频组件样式 */
    audio {
        width: 100% !important;
        border-radius: 10px !important;
    }
    
    /* 滑块样式 */
    input[type="range"] {
        background: linear-gradient(to right, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* 标题样式 */
    h1, h2, h3 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
"""

# 创建Gradio界面
with gr.Blocks(title="IndexTTS Demo", theme=gr.themes.Soft(), css=custom_css) as demo:
    mutex = threading.Lock()
    
    # 顶部导航栏
    with gr.Row():
        gr.HTML('''
        <div style="text-align: center; padding: 20px; background: white; border-radius: 15px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h1 style="margin: 0; font-size: 2.5em;">🎙️ IndexTTS 2.0</h1>
            <p style="color: #666; margin: 10px 0;">Emotionally Expressive Zero-Shot Text-to-Speech System</p>
            <div style="display: flex; justify-content: center; gap: 20px; margin-top: 15px;">
                <a href='https://arxiv.org/abs/2506.21619' target='_blank' style='text-decoration: none;'>
                    <img src='https://img.shields.io/badge/ArXiv-2506.21619-red' style='height: 25px;'>
                </a>
                <a href='#' style='text-decoration: none;'>
                    <img src='https://img.shields.io/badge/Version-2.0-blue' style='height: 25px;'>
                </a>
                <a href='#' style='text-decoration: none;'>
                    <img src='https://img.shields.io/badge/License-MIT-green' style='height: 25px;'>
                </a>
            </div>
        </div>
        ''')
    
    # 主选项卡
    with gr.Tabs(elem_classes="tabs"):
        # 🎵 音频生成选项卡
        with gr.Tab("🎵 音频生成", elem_id="generation_tab"):
            # 实时状态监控面板
            with gr.Row():
                with gr.Column(scale=3):
                    queue_status_display = gr.HTML(value="<div style='padding: 10px;'>初始化中...</div>")
                with gr.Column(scale=1):
                    with gr.Row():
                        refresh_queue_btn = gr.Button("🔄 刷新", size="sm", elem_classes="primary-btn")
                        clear_queue_btn = gr.Button("🗑️ 清理", size="sm")
            
            # 任务队列表格
            queue_table = gr.Dataframe(
                headers=["序号", "文本预览", "状态", "提交时间"],
                label="📋 任务队列",
                interactive=False,
                elem_classes="card"
            )
            
            # 主要输入区域
            with gr.Row():
                with gr.Column(scale=1):
                    # 音色选择卡片
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### 🎨 音色选择")
                        prompt_audio = gr.Dropdown(
                            label="选择音色",
                            choices=timbre_choices_boot,
                            value=default_timbre_boot,
                            interactive=True,
                        )
                        refresh_timbres_btn = gr.Button("刷新列表", size="sm")
                        timbre_preview = gr.Audio(
                            label="试听",
                            value=default_timbre_boot,
                            visible=bool(default_timbre_boot),
                            autoplay=False
                        )
                
                with gr.Column(scale=2):
                    # 文本输入卡片
                    with gr.Group(elem_classes="card"):
                        gr.Markdown("### ✍️ 输入文本")
                        input_text_single = gr.TextArea(
                            placeholder="请输入要转换的文本内容...",
                            lines=5,
                            info=f"模型版本: {tts.model_version or '1.0'}"
                        )
                        gen_button = gr.Button(
                            "🚀 添加到生成队列", 
                            variant="primary", 
                            size="lg",
                            elem_classes="primary-btn"
                        )
            
            # 生成结果展示区
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🎧 最新生成")
                    output_audio = gr.Audio(label="当前结果", visible=True)
                
                with gr.Column():
                    with gr.Row():
                        gr.Markdown("### 📚 历史记录")
                        refresh_history_btn = gr.Button("刷新", size="sm")
                    with gr.Row():
                        history_audio_1 = gr.Audio(label="最近 1", visible=False)
                        history_audio_2 = gr.Audio(label="最近 2", visible=False)
                    with gr.Row():
                        history_audio_3 = gr.Audio(label="最近 3", visible=False)
                        history_audio_4 = gr.Audio(label="最近 4", visible=False)
                    with gr.Row():
                        history_audio_5 = gr.Audio(label="最近 5", visible=False)
                        history_audio_6 = gr.Audio(label="最近 6", visible=False)
        
        # ⚙️ 高级设置选项卡
        with gr.Tab("⚙️ 高级设置", elem_id="settings_tab"):
            # 情感控制设置
            with gr.Group(elem_classes="card"):
                gr.Markdown("### 🎭 情感控制")
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES,
                    type="index",
                    value=EMO_CHOICES[0],
                    label="控制方式"
                )
                
                # 情感参考音频
                with gr.Group(visible=False) as emotion_reference_group:
                    emo_upload = gr.Audio(label="上传情感参考音频", type="filepath")
                    emo_weight = gr.Slider(label="情感权重", minimum=0.0, maximum=1.6, value=0.8, step=0.01)
                
                # 情感随机采样
                emo_random = gr.Checkbox(label="启用情感随机采样", value=False, visible=False)
                
                # 情感向量控制
                with gr.Group(visible=False) as emotion_vector_group:
                    gr.Markdown("#### 情感向量调节")
                    with gr.Row():
                        with gr.Column():
                            vec1 = gr.Slider(label="😊 喜", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                            vec2 = gr.Slider(label="😠 怒", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                            vec3 = gr.Slider(label="😢 哀", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                            vec4 = gr.Slider(label="😨 惧", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                        with gr.Column():
                            vec5 = gr.Slider(label="🤢 厌恶", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                            vec6 = gr.Slider(label="😔 低落", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                            vec7 = gr.Slider(label="😲 惊喜", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                            vec8 = gr.Slider(label="😌 平静", minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                
                # 情感文本描述
                with gr.Group(visible=False) as emo_text_group:
                    emo_text = gr.Textbox(
                        label="情感描述",
                        placeholder="输入情绪描述（如：高兴、愤怒、悲伤等）",
                        value=""
                    )
            
            # 生成参数设置
            with gr.Group(elem_classes="card"):
                gr.Markdown("### 🔧 生成参数")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### GPT2 采样参数")
                        do_sample = gr.Checkbox(label="启用采样", value=True)
                        temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                        top_p = gr.Slider(label="Top P", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="Top K", minimum=0, maximum=100, value=30, step=1)
                    
                    with gr.Column():
                        gr.Markdown("#### 生成控制")
                        num_beams = gr.Slider(label="Beam数量", value=3, minimum=1, maximum=10, step=1)
                        repetition_penalty = gr.Number(label="重复惩罚", value=10.0, minimum=0.1, maximum=20.0)
                        length_penalty = gr.Number(label="长度惩罚", value=0.0, minimum=-2.0, maximum=2.0)
                        max_mel_tokens = gr.Slider(label="最大Token数", value=1500, minimum=50, maximum=3000, step=10)
            
            # 分句设置
            with gr.Group(elem_classes="card"):
                gr.Markdown("### 📝 分句设置")
                max_text_tokens_per_segment = gr.Slider(
                    label="分句最大Token数",
                    value=200,
                    minimum=20,
                    maximum=500,
                    step=2,
                    info="建议80-200，影响音频质量和生成速度"
                )
                segments_preview = gr.Dataframe(
                    headers=["序号", "分句内容", "Token数"],
                    label="分句预览",
                    wrap=True
                )
        
        # 📖 使用示例选项卡
        with gr.Tab("📖 使用示例", elem_id="examples_tab"):
            if len(example_cases) > 0:
                gr.Examples(
                    examples=example_cases,
                    examples_per_page=10,
                    inputs=[prompt_audio,
                            emo_control_method,
                            input_text_single,
                            emo_upload,
                            emo_weight,
                            emo_text,
                            vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8],
                    label="点击示例快速体验"
                )
        
        # ℹ️ 关于选项卡
        with gr.Tab("ℹ️ 关于", elem_id="about_tab"):
            gr.Markdown("""
            ## 关于 IndexTTS 2.0
            
            IndexTTS 是一个先进的零样本文本转语音系统，具有以下特点：
            
            ### ✨ 主要特性
            - 🎭 **情感表达**：支持多种情感控制方式
            - 🎨 **音色克隆**：仅需几秒参考音频即可克隆音色
            - ⚡ **高效生成**：优化的推理引擎，快速生成高质量音频
            - 🌏 **多语言支持**：支持中文和英文
            
            ### 📚 使用指南
            1. **选择音色**：从预设音色中选择或上传自定义音频
            2. **输入文本**：输入要转换的文本内容
            3. **调整参数**：根据需要调整情感和生成参数
            4. **生成音频**：点击生成按钮，等待处理完成
            
            ### 🔗 相关链接
            - [论文地址](https://arxiv.org/abs/2506.21619)
            - [GitHub仓库](#)
            - [模型下载](#)
            
            ### 📧 联系我们
            如有问题或建议，请通过以下方式联系：
            - Email: example@email.com
            - Issue: GitHub Issues
            
            ---
            *© 2024 IndexTTS Team. All rights reserved.*
            """)
    
    # 高级参数列表（用于传递）
    advanced_params = [
        do_sample, top_p, top_k, temperature,
        length_penalty, num_beams, repetition_penalty, max_mel_tokens,
    ]
    
    # 事件绑定
    emo_control_method.select(
        on_method_select,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group, emo_random, emotion_vector_group, emo_text_group]
    )
    
    input_text_single.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )
    
    max_text_tokens_per_segment.change(
        on_input_text_change,
        inputs=[input_text_single, max_text_tokens_per_segment],
        outputs=[segments_preview]
    )
    
    prompt_audio.change(
        update_timbre_preview,
        inputs=[prompt_audio],
        outputs=[timbre_preview]
    )
    
    refresh_timbres_btn.click(
        refresh_timbres,
        inputs=[],
        outputs=[prompt_audio, timbre_preview]
    )
    
    gen_button.click(
        add_to_queue,
        inputs=[emo_control_method, prompt_audio, input_text_single, emo_upload, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                emo_text, emo_random,
                max_text_tokens_per_segment,
                *advanced_params],
        outputs=[queue_status_display, queue_table, output_audio, 
                 history_audio_1, history_audio_2, history_audio_3,
                 history_audio_4, history_audio_5, history_audio_6]
    )
    
    refresh_queue_btn.click(
        get_queue_status,
        inputs=[],
        outputs=[queue_status_display, queue_table, output_audio,
                 history_audio_1, history_audio_2, history_audio_3,
                 history_audio_4, history_audio_5, history_audio_6]
    )
    
    clear_queue_btn.click(
        clear_completed_tasks,
        inputs=[],
        outputs=[queue_status_display, queue_table, output_audio,
                 history_audio_1, history_audio_2, history_audio_3,
                 history_audio_4, history_audio_5, history_audio_6]
    )
    
    refresh_history_btn.click(
        refresh_history,
        inputs=[],
        outputs=[history_audio_1, history_audio_2, history_audio_3,
                 history_audio_4, history_audio_5, history_audio_6]
    )
    
    # 自动刷新队列状态
    demo.load(
        continuous_queue_refresh,
        inputs=[],
        outputs=[queue_status_display, queue_table, output_audio,
                 history_audio_1, history_audio_2, history_audio_3,
                 history_audio_4, history_audio_5, history_audio_6],
        show_progress="hidden"
    )

if __name__ == "__main__":
    # 启用队列功能
    demo.queue(
        max_size=50,
        default_concurrency_limit=1
    )
    
    demo.launch(
        server_name=cmd_args.host,
        server_port=cmd_args.port,
        share=False,
        favicon_path=None,  # 可以添加自定义图标
        show_error=True
    )
