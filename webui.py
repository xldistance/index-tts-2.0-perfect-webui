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
parser = argparse.ArgumentParser(description="IndexTTS WebUI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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

for file in ["bpe.model", "gpt.pth", "config.yaml", "s2mel.pth", "wav2vec2bert_stats.pt"]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

import gradio as gr
from indextts.infer_v2 import IndexTTS2
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="Auto")
MODE = 'local'
tts = IndexTTS2(model_dir=cmd_args.model_dir, cfg_path=os.path.join(cmd_args.model_dir, "config.yaml"),
                use_fp16=cmd_args.fp16, use_deepspeed=cmd_args.use_deepspeed, use_cuda_kernel=cmd_args.cuda_kernel)

LANGUAGES = {"中文": "zh_CN", "English": "en_US"}
EMO_CHOICES = [i18n("与音色参考音频相同"), i18n("使用情感参考音频"), i18n("使用情感向量控制"), i18n("使用情感描述文本控制")]

os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("prompts", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

playback_positions = {}
playback_lock = threading.Lock()
SAVED_TIMBRES_DIR = os.path.join(current_dir, "saved_timbres")
os.makedirs(SAVED_TIMBRES_DIR, exist_ok=True)
SUPPORTED_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")

generation_history = deque(maxlen=10)
generation_lock = threading.Lock()

task_queue = queue.Queue()
queue_status = {}
queue_lock = threading.Lock()
processing_thread = None
stop_processing = False
current_task_id = None

class TaskStatus:
    PENDING = "等待中"
    PROCESSING = "生成中"
    COMPLETED = "已完成"
    FAILED = "失败"
    CANCELLED = "已取消"

def list_timbres():
    files = []
    if os.path.isdir(SAVED_TIMBRES_DIR):
        for fn in os.listdir(SAVED_TIMBRES_DIR):
            if fn.lower().endswith(SUPPORTED_AUDIO_EXTS):
                files.append(os.path.join(SAVED_TIMBRES_DIR, fn))
    files.sort(key=lambda p: os.path.basename(p).lower())
    return files

def get_default_timbre():
    timbres = list_timbres()
    sweet_voice_path = os.path.join(SAVED_TIMBRES_DIR, "甜美女声1.mp3")
    if sweet_voice_path in timbres:
        return sweet_voice_path
    return timbres[0] if timbres else None

timbre_choices_boot = list_timbres()
default_timbre_boot = get_default_timbre()
MAX_LENGTH_TO_USE_SPEED = 70

with open("examples/cases.jsonl", "r", encoding="utf-8") as f:
    example_cases = []
    for line in f:
        line = line.strip()
        if not line:
            continue
        example = json.loads(line)
        emo_audio_path = os.path.join("examples", example["emo_audio"]) if example.get("emo_audio") else None
        example_cases.append([
            default_timbre_boot, EMO_CHOICES[example.get("emo_mode", 0)], example.get("text"), emo_audio_path,
            example.get("emo_weight", 1.0), example.get("emo_text", ""),
            example.get("emo_vec_1", 0), example.get("emo_vec_2", 0), example.get("emo_vec_3", 0), example.get("emo_vec_4", 0),
            example.get("emo_vec_5", 0), example.get("emo_vec_6", 0), example.get("emo_vec_7", 0), example.get("emo_vec_8", 0)
        ])

def add_to_history(audio_path, text=""):
    display_text = text[:50] if text else "未命名音频"
    with generation_lock:
        generation_history.append({'path': audio_path, 'time': datetime.now(), 'text': display_text})

def get_history_display():
    with generation_lock:
        if not generation_history:
            return ([None] * 6, [gr.update(value=0)] * 6, [gr.update(label=f"最近 {i+1}") for i in range(6)])
        history_list = list(generation_history)
        history_list.reverse()
        audio_paths, position_states, labels = [None] * 6, [0] * 6, [f"最近 {i+1}" for i in range(6)]
        for i, item in enumerate(history_list[:6]):
            path = item['path']
            audio_paths[i] = path
            with playback_lock:
                position_states[i] = playback_positions.get(path, 0)
            labels[i] = item.get('text', f"最近 {i+1}")
        return audio_paths, [gr.update(value=pos) for pos in position_states], [gr.update(label=lbl) for lbl in labels]

def refresh_history():
    audio_paths, _, label_updates = get_history_display()
    audio_updates = [gr.update(value=h, visible=h is not None) for h in audio_paths]
    return (*audio_updates, *label_updates)

def process_queue():
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
                    add_to_history(output, task['params']['text'])
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
    emo_control_method, prompt, text = params['emo_control_method'], params['prompt'], params['text']
    emo_ref_path, emo_weight = params['emo_ref_path'], params['emo_weight']
    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8 = params['vec']
    emo_text, emo_random = params['emo_text'], params['emo_random']
    max_text_tokens_per_segment, kwargs = params['max_text_tokens_per_segment'], params['kwargs']
    
    timestamp = int(time.time() * 1000)
    cleaned_text = re.sub(r'[\n "\' :]', ' ', text)
    output_path = os.path.join("outputs", f"{Path(prompt).stem}_{cleaned_text[:50]}_{timestamp}.wav")
    
    if type(emo_control_method) is not int:
        emo_control_method = emo_control_method.value
    if emo_control_method == 0:
        emo_ref_path, emo_weight = None, 1.0
    vec = None
    if emo_control_method == 2:
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        if sum(vec) > 1.5:
            raise ValueError(i18n("情感向量之和不能超过1.5，请调整后重试。"))
    if emo_text == "":
        emo_text = None
    print(f"Emo control mode:{emo_control_method}, vec:{vec}")
    
    return tts.infer(spk_audio_prompt=prompt, text=text, output_path=output_path,
                     emo_audio_prompt=emo_ref_path, emo_alpha=emo_weight, emo_vector=vec,
                     use_emo_text=(emo_control_method == 3), emo_text=emo_text, use_random=emo_random,
                     verbose=cmd_args.verbose, max_text_tokens_per_segment=int(max_text_tokens_per_segment), **kwargs)

def get_queue_counters():
    """只获取计数器数值（返回字符串）"""
    with queue_lock:
        pending = sum(1 for s in queue_status.values() if s['status'] == TaskStatus.PENDING)
        processing = sum(1 for s in queue_status.values() if s['status'] == TaskStatus.PROCESSING)
        completed = sum(1 for s in queue_status.values() if s['status'] == TaskStatus.COMPLETED)
        queue_len = task_queue.qsize()
    return str(processing), str(pending), str(completed), str(queue_len)

def add_to_queue(emo_control_method, prompt, text, emo_ref_path, emo_weight,
                 vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, emo_text, emo_random,
                 max_text_tokens_per_segment, *args):
    global processing_thread, stop_processing
    
    processing, pending, completed, queue_len = get_queue_counters()
    
    if not text or text.strip() == "":
        return (processing, pending, completed, queue_len, gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    if not prompt:
        return (processing, pending, completed, queue_len, gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())
    
    do_sample, top_p, top_k, temperature, length_penalty, num_beams, repetition_penalty, max_mel_tokens = args
    kwargs = {"do_sample": bool(do_sample), "top_p": float(top_p), "top_k": int(top_k) if int(top_k) > 0 else None,
              "temperature": float(temperature), "length_penalty": float(length_penalty), "num_beams": int(num_beams),
              "repetition_penalty": float(repetition_penalty), "max_mel_tokens": int(max_mel_tokens)}
    
    task_id = str(uuid.uuid4())
    task = {'id': task_id, 'params': {
        'emo_control_method': emo_control_method, 'prompt': prompt, 'text': text.strip(),
        'emo_ref_path': emo_ref_path if emo_ref_path else None, 'emo_weight': float(emo_weight) if emo_weight else 0.8,
        'vec': [float(v) for v in [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]],
        'emo_text': emo_text if emo_text else None, 'emo_random': bool(emo_random),
        'max_text_tokens_per_segment': int(max_text_tokens_per_segment), 'kwargs': kwargs
    }}
    
    with queue_lock:
        queue_status[task_id] = {'status': TaskStatus.PENDING, 'text': text[:80] + '...' if len(text) > 80 else text,
                                  'submit_time': datetime.now(), 'position': task_queue.qsize() + 1}
    task_queue.put(task)
    
    if processing_thread is None or not processing_thread.is_alive():
        stop_processing = False
        processing_thread = threading.Thread(target=process_queue, daemon=True)
        processing_thread.start()
    
    return get_full_status_update()

def get_full_status_update():
    """返回完整状态更新（计数器 + 表格 + 音频）"""
    processing, pending, completed, queue_len = get_queue_counters()
    
    with queue_lock:
        data = []
        sorted_tasks = sorted(queue_status.items(), key=lambda x: x[1]['submit_time'])
        for idx, (task_id, info) in enumerate(sorted_tasks[-10:], 1):
            emoji = {"等待中": "⏳", "生成中": "🔄", "已完成": "✅", "失败": "❌", "已取消": "🚫"}.get(info['status'], "")
            data.append([idx, info['text'], f"{emoji} {info['status']}", info['submit_time'].strftime("%H:%M:%S")])
        
        latest_output = None
        for task_id, info in reversed(sorted_tasks):
            if info['status'] == TaskStatus.COMPLETED and 'output' in info:
                latest_output = info['output']
                break
    
    audio_paths, _, label_updates = get_history_display()
    history_updates = [gr.update(value=ap, visible=bool(ap)) for ap in audio_paths]
    
    # 计数器已经是字符串格式（来自 get_queue_counters）
    return (processing, pending, completed, queue_len,
            gr.update(value=data),
            gr.update(value=latest_output, visible=bool(latest_output)) if latest_output else gr.update(),
            *history_updates, *label_updates)


def refresh_counters_and_audio():
    """刷新计数器和最新音频（用于定时器）- 返回5个值"""
    # 获取计数器
    with queue_lock:
        pending = sum(1 for s in queue_status.values() if s['status'] == TaskStatus.PENDING)
        processing = sum(1 for s in queue_status.values() if s['status'] == TaskStatus.PROCESSING)
        completed = sum(1 for s in queue_status.values() if s['status'] == TaskStatus.COMPLETED)
        queue_len = task_queue.qsize()
        
        # 获取最新完成的音频
        latest_output = None
        sorted_tasks = sorted(queue_status.items(), key=lambda x: x[1]['submit_time'])
        for task_id, info in reversed(sorted_tasks):
            if info['status'] == TaskStatus.COMPLETED and 'output' in info:
                latest_output = info['output']
                break
    
    # 返回5个值：4个计数器字符串 + 1个音频更新
    if latest_output:
        return str(processing), str(pending), str(completed), str(queue_len), gr.update(value=latest_output, visible=True)
    else:
        return str(processing), str(pending), str(completed), str(queue_len), gr.update()

def clear_completed_tasks():
    with queue_lock:
        to_remove = [tid for tid, info in queue_status.items() 
                    if info['status'] in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]]
        for tid in to_remove:
            del queue_status[tid]
    return get_full_status_update()

def refresh_counters_only():
    """只刷新计数器（用于定时器）"""
    return get_queue_counters()

def on_input_text_change(text, max_text_tokens_per_segment):
    if text and len(text) > 0:
        text_tokens_list = tts.tokenizer.tokenize(text)
        segments = tts.tokenizer.split_segments(text_tokens_list, max_text_tokens_per_segment=int(max_text_tokens_per_segment))
        return gr.update(value=[[i, ''.join(s), len(s)] for i, s in enumerate(segments)], visible=True)
    return gr.update(value=[])

def on_method_select(emo_control_method):
    vis = [(False, False, False, False), (True, False, False, False), (False, True, True, False), (False, True, False, True)]
    v = vis[emo_control_method] if emo_control_method < 4 else vis[0]
    return tuple(gr.update(visible=x) for x in v)

def update_timbre_preview(selected_path):
    if selected_path and os.path.exists(selected_path):
        return gr.update(value=selected_path, visible=True)
    return gr.update(value=None, visible=False)

def refresh_timbres():
    choices, value = list_timbres(), get_default_timbre()
    return gr.update(choices=choices, value=value), gr.update(value=value, visible=bool(value))

# Solarized Light 配色方案
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --base03: #002b36;
    --base02: #073642;
    --base01: #586e75;
    --base00: #657b83;
    --base0: #839496;
    --base1: #93a1a1;
    --base2: #eee8d5;
    --base3: #fdf6e3;
    --yellow: #b58900;
    --orange: #cb4b16;
    --red: #dc322f;
    --magenta: #d33682;
    --violet: #6c71c4;
    --blue: #268bd2;
    --cyan: #2aa198;
    --green: #859900;
}

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

.gradio-container {
    background: var(--base3) !important;
    min-height: 100vh;
}

.main-header {
    background: var(--base2);
    border: 1px solid var(--base1);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

.main-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--blue);
    margin: 0 0 8px 0;
}

.main-header p { color: var(--base00); font-size: 1rem; margin: 0; }

/* 静态状态面板样式 */
.status-panel-static {
    background: #eee8d5;
    border: 1px solid #93a1a1;
    border-radius: 16px 16px 0 0;
    padding: 20px 20px 8px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.status-grid-static {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}

.status-card-static {
    background: #fdf6e3;
    border-radius: 12px 12px 0 0;
    padding: 16px 16px 8px 16px;
    text-align: center;
    border: 1px solid #93a1a1;
    border-bottom: none;
}

.status-icon-static { font-size: 1.5rem; margin-bottom: 6px; }
.status-label-static { color: #657b83; font-size: 0.8rem; font-weight: 500; }

/* 计数器父行样式 */
.status-counter-row {
    background: #eee8d5 !important;
    border: 1px solid #93a1a1 !important;
    border-top: none !important;
    border-radius: 0 0 16px 16px !important;
    padding: 0 20px 20px 20px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
    gap: 12px !important;
}

/* 计数器单元格样式 - 加强选择器优先级 */
#counter-processing, #counter-pending, #counter-completed, #counter-queue {
    background: #fdf6e3 !important;
    border: 1px solid #93a1a1 !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 12px 16px 20px 16px !important;
    text-align: center !important;
    flex: 1 !important;
}

/* 隐藏 Gradio 容器的额外元素 */
#counter-processing > div,
#counter-pending > div,
#counter-completed > div,
#counter-queue > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* 覆盖 Gradio 默认输入框样式 */
#counter-processing textarea,
#counter-processing input[type="text"],
#counter-processing input,
#counter-pending textarea,
#counter-pending input[type="text"],
#counter-pending input,
#counter-completed textarea,
#counter-completed input[type="text"],
#counter-completed input,
#counter-queue textarea,
#counter-queue input[type="text"],
#counter-queue input,
#counter-processing .wrap input,
#counter-pending .wrap input,
#counter-completed .wrap input,
#counter-queue .wrap input,
.status-counter-row input,
.status-counter-row textarea {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    text-align: center !important;
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    height: auto !important;
    min-height: auto !important;
    box-shadow: none !important;
    width: 100% !important;
    line-height: 1.2 !important;
    letter-spacing: -1px !important;
    outline: none !important;
}

#counter-processing textarea, #counter-processing input { 
    color: #cb4b16 !important; 
    text-shadow: 0 1px 2px rgba(203, 75, 22, 0.2) !important; 
}
#counter-pending textarea, #counter-pending input { 
    color: #268bd2 !important; 
    text-shadow: 0 1px 2px rgba(38, 139, 210, 0.2) !important; 
}
#counter-completed textarea, #counter-completed input { 
    color: #859900 !important; 
    text-shadow: 0 1px 2px rgba(133, 153, 0, 0.2) !important; 
}
#counter-queue textarea, #counter-queue input { 
    color: #6c71c4 !important; 
    text-shadow: 0 1px 2px rgba(108, 113, 196, 0.2) !important; 
}

#counter-processing textarea:focus,
#counter-processing input:focus,
#counter-pending textarea:focus,
#counter-pending input:focus,
#counter-completed textarea:focus,
#counter-completed input:focus,
#counter-queue textarea:focus,
#counter-queue input:focus,
.status-counter-row input:focus,
.status-counter-row textarea:focus {
    outline: none !important;
    box-shadow: none !important;
    border: none !important;
}

/* 移除 Gradio 输入框的边框和背景 */
.status-counter-row .gr-box,
.status-counter-row .gr-input,
.status-counter-row .gr-text-input,
#counter-processing .gr-box,
#counter-pending .gr-box,
#counter-completed .gr-box,
#counter-queue .gr-box {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.glass-card {
    background: var(--base2) !important;
    border: 1px solid var(--base1) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}

.section-title {
    color: var(--base01);
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.gen-btn {
    background: linear-gradient(135deg, var(--blue) 0%, var(--cyan) 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: white !important;
    cursor: pointer;
    transition: all 0.2s ease !important;
    box-shadow: 0 3px 12px rgba(38, 139, 210, 0.3) !important;
}

.gen-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(38, 139, 210, 0.4) !important;
}

.small-btn {
    background: var(--base3) !important;
    border: 1px solid var(--base1) !important;
    border-radius: 8px !important;
    color: var(--base01) !important;
    font-size: 0.85rem !important;
    padding: 6px 12px !important;
    transition: all 0.2s ease !important;
}

.small-btn:hover {
    background: var(--base2) !important;
    border-color: var(--blue) !important;
    color: var(--blue) !important;
}

.tabs { background: transparent !important; }
.tab-nav { 
    background: var(--base2) !important; 
    border-radius: 12px !important; 
    padding: 6px !important; 
    margin-bottom: 16px !important; 
    border: 1px solid var(--base1) !important;
}
.tab-nav button { 
    background: transparent !important; 
    color: var(--base00) !important; 
    border: none !important; 
    border-radius: 8px !important; 
    padding: 10px 20px !important; 
    font-weight: 500 !important; 
    transition: all 0.2s ease !important; 
}
.tab-nav button.selected { 
    background: var(--base3) !important; 
    color: var(--blue) !important; 
    box-shadow: 0 2px 6px rgba(0,0,0,0.08) !important;
}
.tab-nav button:hover:not(.selected) { 
    background: rgba(38, 139, 210, 0.1) !important; 
}

input, textarea, select {
    background: var(--base3) !important;
    border: 1px solid var(--base1) !important;
    border-radius: 10px !important;
    color: var(--base01) !important;
    transition: all 0.2s ease !important;
}

input:focus, textarea:focus, select:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(38, 139, 210, 0.15) !important;
}

label { color: var(--base01) !important; font-weight: 500 !important; }
.info { color: var(--base00) !important; }

input[type="range"] {
    background: var(--base2) !important;
    border-radius: 6px !important;
}

input[type="range"]::-webkit-slider-thumb {
    background: var(--blue) !important;
    border: none !important;
}

.dataframe { 
    background: var(--base3) !important; 
    border-radius: 12px !important; 
    overflow: hidden !important; 
    border: 1px solid var(--base1) !important;
}
.dataframe thead { background: var(--base2) !important; }
.dataframe th { 
    color: var(--base01) !important; 
    font-weight: 600 !important; 
    padding: 12px !important; 
    border: none !important; 
}
.dataframe td { 
    color: var(--base00) !important; 
    padding: 10px 12px !important; 
    border-bottom: 1px solid var(--base2) !important; 
}
.dataframe tr:hover td { background: var(--base2) !important; }

audio { 
    border-radius: 10px !important; 
    background: var(--base2) !important; 
}

.badge-row { display: flex; gap: 10px; justify-content: center; margin-top: 14px; }
.badge-row a { transition: transform 0.2s ease; }
.badge-row a:hover { transform: scale(1.05); }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--base2); border-radius: 4px; }
::-webkit-scrollbar-thumb { background: var(--base1); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--base00); }

/* Gradio 组件覆盖 */
.gr-button { border-radius: 10px !important; }
.gr-box { border-radius: 12px !important; border-color: var(--base1) !important; }
.gr-input { border-radius: 10px !important; }
.gr-panel { background: var(--base2) !important; border-radius: 12px !important; }
"""

with gr.Blocks(title="IndexTTS 2.0", theme=gr.themes.Soft(), css=custom_css) as demo:
    mutex = threading.Lock()
    
    gr.HTML('''
    <div class="main-header">
        <h1>🎙️ IndexTTS 2.0</h1>
        <p>Emotionally Expressive Zero-Shot Text-to-Speech System</p>
        <div class="badge-row">
            <a href='https://arxiv.org/abs/2506.21619' target='_blank'><img src='https://img.shields.io/badge/ArXiv-2506.21619-dc322f?style=for-the-badge' height='26'></a>
            <a href='#'><img src='https://img.shields.io/badge/Version-2.0-268bd2?style=for-the-badge' height='26'></a>
            <a href='#'><img src='https://img.shields.io/badge/License-MIT-859900?style=for-the-badge' height='26'></a>
        </div>
    </div>
    ''')
    
    # 状态面板 - 静态HTML结构 + 动态计数器组件
    gr.HTML('''
    <div class="status-panel-static">
        <div class="status-grid-static">
            <div class="status-card-static">
                <div class="status-icon-static">🔄</div>
                <div class="status-label-static">处理中</div>
            </div>
            <div class="status-card-static">
                <div class="status-icon-static">⏳</div>
                <div class="status-label-static">等待中</div>
            </div>
            <div class="status-card-static">
                <div class="status-icon-static">✅</div>
                <div class="status-label-static">已完成</div>
            </div>
            <div class="status-card-static">
                <div class="status-icon-static">📁</div>
                <div class="status-label-static">队列长度</div>
            </div>
        </div>
    </div>
    ''')
    
    # 计数器数字 - 使用 Textbox 组件
    with gr.Row(elem_classes="status-counter-row"):
        counter_processing = gr.Textbox(value="0", show_label=False, container=False, elem_id="counter-processing", interactive=False)
        counter_pending = gr.Textbox(value="0", show_label=False, container=False, elem_id="counter-pending", interactive=False)
        counter_completed = gr.Textbox(value="0", show_label=False, container=False, elem_id="counter-completed", interactive=False)
        counter_queue = gr.Textbox(value="0", show_label=False, container=False, elem_id="counter-queue", interactive=False)
    
    with gr.Tabs():
        with gr.Tab("🎵 音频生成"):
            with gr.Row():
                with gr.Column(scale=1):
                    refresh_queue_btn = gr.Button("🔄 刷新队列", elem_classes="small-btn")
                    clear_queue_btn = gr.Button("🗑️ 清理已完成", elem_classes="small-btn")
            
            queue_table = gr.Dataframe(headers=["序号", "文本预览", "状态", "提交时间"], label="📋 任务队列", interactive=False)
            
            with gr.Row():
                with gr.Column(scale=1, elem_classes="glass-card"):
                    gr.HTML('<div class="section-title">🎨 音色选择</div>')
                    prompt_audio = gr.Dropdown(label="选择音色", choices=timbre_choices_boot, value=default_timbre_boot, interactive=True)
                    refresh_timbres_btn = gr.Button("刷新列表", size="sm", elem_classes="small-btn")
                    timbre_preview = gr.Audio(label="试听", value=default_timbre_boot, visible=bool(default_timbre_boot), autoplay=False)
                
                with gr.Column(scale=2, elem_classes="glass-card"):
                    gr.HTML('<div class="section-title">✍️ 输入文本</div>')
                    input_text_single = gr.TextArea(placeholder="请输入要转换的文本内容...", lines=5, info=f"模型版本: {tts.model_version or '1.0'}")
                    gen_button = gr.Button("🚀 添加到生成队列", variant="primary", size="lg", elem_classes="gen-btn")
            
            with gr.Row():
                with gr.Column(elem_classes="glass-card"):
                    gr.HTML('<div class="section-title">🎧 最新生成</div>')
                    output_audio = gr.Audio(label="当前结果", visible=True)
                
                with gr.Column(elem_classes="glass-card"):
                    with gr.Row():
                        gr.HTML('<div class="section-title">📚 历史记录</div>')
                        refresh_history_btn = gr.Button("刷新", size="sm", elem_classes="small-btn")
                    history_audio_1 = gr.Audio(label="最近 1", visible=False)
                    history_audio_2 = gr.Audio(label="最近 2", visible=False)
                    history_audio_3 = gr.Audio(label="最近 3", visible=False)
                    history_audio_4 = gr.Audio(label="最近 4", visible=False)
                    history_audio_5 = gr.Audio(label="最近 5", visible=False)
                    history_audio_6 = gr.Audio(label="最近 6", visible=False)
        
        with gr.Tab("⚙️ 高级设置"):
            with gr.Group(elem_classes="glass-card"):
                gr.HTML('<div class="section-title">🎭 情感控制</div>')
                emo_control_method = gr.Radio(choices=EMO_CHOICES, type="index", value=EMO_CHOICES[0], label="控制方式")
                
                with gr.Group(visible=False) as emotion_reference_group:
                    emo_upload = gr.Audio(label="上传情感参考音频", type="filepath")
                    emo_weight = gr.Slider(label="情感权重", minimum=0.0, maximum=1.6, value=0.8, step=0.01)
                
                emo_random = gr.Checkbox(label="启用情感随机采样", value=False, visible=False)
                
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
                
                with gr.Group(visible=False) as emo_text_group:
                    emo_text = gr.Textbox(label="情感描述", placeholder="输入情绪描述（如：高兴、愤怒、悲伤等）", value="")
            
            with gr.Group(elem_classes="glass-card"):
                gr.HTML('<div class="section-title">🔧 生成参数</div>')
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
            
            with gr.Group(elem_classes="glass-card"):
                gr.HTML('<div class="section-title">📝 分句设置</div>')
                max_text_tokens_per_segment = gr.Slider(label="分句最大Token数", value=200, minimum=20, maximum=500, step=2, info="建议80-200")
                segments_preview = gr.Dataframe(headers=["序号", "分句内容", "Token数"], label="分句预览", wrap=True)
        
        with gr.Tab("📖 使用示例"):
            if example_cases:
                gr.Examples(examples=example_cases, examples_per_page=10,
                           inputs=[prompt_audio, emo_control_method, input_text_single, emo_upload, emo_weight, emo_text,
                                  vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8], label="点击示例快速体验")
        
        with gr.Tab("ℹ️ 关于"):
            gr.Markdown("""
            ## 关于 IndexTTS 2.0
            
            IndexTTS 是一个先进的零样本文本转语音系统。
            
            ### ✨ 主要特性
            - 🎭 **情感表达**：支持多种情感控制方式
            - 🎨 **音色克隆**：仅需几秒参考音频即可克隆音色
            - ⚡ **高效生成**：优化的推理引擎
            - 🌏 **多语言支持**：支持中文和英文
            
            ### 🔗 相关链接
            - [论文地址](https://arxiv.org/abs/2506.21619)
            """)
    
    advanced_params = [do_sample, top_p, top_k, temperature, length_penalty, num_beams, repetition_penalty, max_mel_tokens]
    
    advanced_params = [do_sample, top_p, top_k, temperature, length_penalty, num_beams, repetition_penalty, max_mel_tokens]
    
    # 事件绑定
    emo_control_method.select(on_method_select, inputs=[emo_control_method], outputs=[emotion_reference_group, emo_random, emotion_vector_group, emo_text_group])
    input_text_single.change(on_input_text_change, inputs=[input_text_single, max_text_tokens_per_segment], outputs=[segments_preview])
    max_text_tokens_per_segment.change(on_input_text_change, inputs=[input_text_single, max_text_tokens_per_segment], outputs=[segments_preview])
    prompt_audio.change(update_timbre_preview, inputs=[prompt_audio], outputs=[timbre_preview])
    refresh_timbres_btn.click(refresh_timbres, inputs=[], outputs=[prompt_audio, timbre_preview])
    
    # 生成按钮 - 返回完整更新
    gen_button.click(add_to_queue,
        inputs=[emo_control_method, prompt_audio, input_text_single, emo_upload, emo_weight,
                vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, emo_text, emo_random,
                max_text_tokens_per_segment, *advanced_params],
        outputs=[counter_processing, counter_pending, counter_completed, counter_queue,
                 queue_table, output_audio,
                 history_audio_1, history_audio_2, history_audio_3, history_audio_4, history_audio_5, history_audio_6,
                 history_audio_1, history_audio_2, history_audio_3, history_audio_4, history_audio_5, history_audio_6])
    
    # 刷新队列按钮 - 返回完整更新
    refresh_queue_btn.click(get_full_status_update, inputs=[],
        outputs=[counter_processing, counter_pending, counter_completed, counter_queue,
                 queue_table, output_audio,
                 history_audio_1, history_audio_2, history_audio_3, history_audio_4, history_audio_5, history_audio_6,
                 history_audio_1, history_audio_2, history_audio_3, history_audio_4, history_audio_5, history_audio_6])
    
    # 清理按钮
    clear_queue_btn.click(clear_completed_tasks, inputs=[],
        outputs=[counter_processing, counter_pending, counter_completed, counter_queue,
                 queue_table, output_audio,
                 history_audio_1, history_audio_2, history_audio_3, history_audio_4, history_audio_5, history_audio_6,
                 history_audio_1, history_audio_2, history_audio_3, history_audio_4, history_audio_5, history_audio_6])
    
    refresh_history_btn.click(refresh_history, inputs=[],
        outputs=[history_audio_1, history_audio_2, history_audio_3, history_audio_4, history_audio_5, history_audio_6,
                 history_audio_1, history_audio_2, history_audio_3, history_audio_4, history_audio_5, history_audio_6])
    
    # 定时器 - 刷新计数器和最新音频
    timer = gr.Timer(value=3, active=True)
    timer.tick(fn=refresh_counters_and_audio, inputs=[], outputs=[counter_processing, counter_pending, counter_completed, counter_queue, output_audio])
    
    # JavaScript: 强制应用计数器样式
    def apply_counter_styles():
        return """
        () => {
            const applyStyles = () => {
                const counterIds = ['counter-processing', 'counter-pending', 'counter-completed', 'counter-queue'];
                const colors = ['#cb4b16', '#268bd2', '#859900', '#6c71c4'];
                
                counterIds.forEach((id, index) => {
                    const container = document.getElementById(id);
                    if (container) {
                        const inputs = container.querySelectorAll('input, textarea');
                        inputs.forEach(input => {
                            input.style.cssText = `
                                font-size: 2.5rem !important;
                                font-weight: 800 !important;
                                text-align: center !important;
                                background: transparent !important;
                                border: none !important;
                                box-shadow: none !important;
                                outline: none !important;
                                color: ${colors[index]} !important;
                                width: 100% !important;
                                padding: 0 !important;
                                line-height: 1.2 !important;
                            `;
                        });
                    }
                });
            };
            
            // 初次应用
            applyStyles();
            
            // 监听DOM变化持续应用
            const observer = new MutationObserver(applyStyles);
            observer.observe(document.body, { childList: true, subtree: true });
            
            // 定期应用以确保样式生效
            setInterval(applyStyles, 1000);
        }
        """
    
    demo.load(fn=None, inputs=None, outputs=None, js=apply_counter_styles())

if __name__ == "__main__":
    demo.queue(max_size=50, default_concurrency_limit=1)
    demo.launch(server_name=cmd_args.host, server_port=cmd_args.port, share=False, show_error=True, inbrowser=True)
