import json
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
import queue
import uuid

import warnings

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
generation_history = deque(maxlen=3)  # 保存最近3个生成结果
generation_lock = threading.Lock()

# ========== 队列系统相关变量 ==========
task_queue = queue.Queue()  # 任务队列
queue_status = {}  # 存储每个任务的状态
queue_lock = threading.Lock()  # 队列状态锁
processing_thread = None  # 处理线程
stop_processing = False  # 停止处理标志
current_task_id = None  # 当前正在处理的任务ID

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
    
    # 如果存在"甜美女声1.mp3"，返回它
    if sweet_voice_path in timbres:
        return sweet_voice_path
    # 否则返回第一个可用的音色
    return timbres[0] if timbres else None

# 预计算下拉默认项（供 Examples 使用）
timbre_choices_boot = list_timbres()
default_timbre_boot = get_default_timbre()

MAX_LENGTH_TO_USE_SPEED = 70

# 读取示例，用 saved_timbres 的默认音色文件做第一列输入
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
            default_timbre_boot,  # 用 saved_timbres 的默认文件作为示例的音色参考
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
            'time': datetime.now()
        })
def continuous_queue_refresh():
    """持续刷新队列状态的生成器函数"""
    while True:
        time.sleep(2)  # 每2秒刷新一次
        yield get_queue_status()
def get_history_display():
    """获取历史记录的显示格式"""
    with generation_lock:
        if not generation_history:
            return None, None, None
        
        # 倒序排列（最新的在前）
        history_list = list(generation_history)
        history_list.reverse()
        
        result = [None, None, None]
        for i, item in enumerate(history_list[:3]):
            if i < 3:
                result[i] = item['path']
        
        return tuple(result)

def refresh_history():
    """刷新历史记录显示（独立函数）"""
    hist1, hist2, hist3 = get_history_display()
    return (
        gr.update(value=hist1, visible=hist1 is not None),
        gr.update(value=hist2, visible=hist2 is not None),
        gr.update(value=hist3, visible=hist3 is not None)
    )

# ========== 队列处理相关函数 ==========
def process_queue():
    """后台处理队列中的任务"""
    global current_task_id, stop_processing
    
    while not stop_processing:
        try:
            # 获取任务（超时1秒）
            task = task_queue.get(timeout=1)
            
            if task is None:  # 停止信号
                break
                
            task_id = task['id']
            current_task_id = task_id
            
            # 更新任务状态为处理中
            with queue_lock:
                if task_id in queue_status:
                    if queue_status[task_id]['status'] == TaskStatus.CANCELLED:
                        continue  # 跳过已取消的任务
                    queue_status[task_id]['status'] = TaskStatus.PROCESSING
                    queue_status[task_id]['start_time'] = datetime.now()
            
            # 执行生成任务
            try:
                output = gen_single_core(task['params'])
                
                # 更新任务状态为完成
                with queue_lock:
                    if task_id in queue_status:
                        queue_status[task_id]['status'] = TaskStatus.COMPLETED
                        queue_status[task_id]['output'] = output
                        queue_status[task_id]['end_time'] = datetime.now()
                        
                # 添加到历史记录
                if output:
                    add_to_history(output)
                    
            except Exception as e:
                # 更新任务状态为失败
                with queue_lock:
                    if task_id in queue_status:
                        queue_status[task_id]['status'] = TaskStatus.FAILED
                        queue_status[task_id]['error'] = str(e)
                        queue_status[task_id]['end_time'] = datetime.now()
            
            current_task_id = None
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Queue processing error: {e}")
            current_task_id = None

def gen_single_core(params):
    """核心生成函数（从原gen_single提取）"""
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
    
    # 生成输出路径
    timestamp = int(time.time() * 1000)
    output_path = os.path.join("outputs", f"spk_{timestamp}.wav")
    
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
    
    # 准备参数
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
    
    # 创建任务
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
    
    # 添加到队列状态
    with queue_lock:
        queue_status[task_id] = {
            'status': TaskStatus.PENDING,
            'text': text[:50] + '...' if len(text) > 50 else text,
            'submit_time': datetime.now(),
            'position': task_queue.qsize() + 1
        }
    
    # 添加到队列
    task_queue.put(task)
    
    # 启动处理线程（如果还没启动）
    if processing_thread is None or not processing_thread.is_alive():
        stop_processing = False
        processing_thread = threading.Thread(target=process_queue, daemon=True)
        processing_thread.start()
    
    return get_queue_status()
def get_queue_status():
    """获取队列状态信息，并同时返回最近3条生成历史的更新值"""
    with queue_lock:
        # 统计各状态任务数
        pending_count = sum(1 for s in queue_status.values() if s['status'] == TaskStatus.PENDING)
        processing_count = sum(1 for s in queue_status.values() if s['status'] == TaskStatus.PROCESSING)
        completed_count = sum(1 for s in queue_status.values() if s['status'] == TaskStatus.COMPLETED)

        # 获取队列信息表格
        data = []
        sorted_tasks = sorted(queue_status.items(), key=lambda x: x[1]['submit_time'])

        for idx, (task_id, info) in enumerate(sorted_tasks[-10:], 1):  # 只显示最近10个
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

        # 创建状态信息
        status_text = f"""
            ### 队列状态
            - 🔄 **正在处理**: {processing_count} 个任务
            - ⏳ **等待中**: {pending_count} 个任务  
            - ✅ **已完成**: {completed_count} 个任务
            - 📊 **队列总长度**: {task_queue.qsize()} 个任务

            **提示**: 任务将按提交时间顺序依次处理
            """

        # 获取最新完成的任务输出（用于 output_audio）
        latest_output = None
        for task_id, info in reversed(sorted_tasks):
            if info['status'] == TaskStatus.COMPLETED and 'output' in info:
                latest_output = info['output']
                break

        # 获取最近三条历史（最新在前）
        hist1, hist2, hist3 = get_history_display()

        # 返回 6 个 gr.update（依次与绑定 outputs 顺序对应）
        queue_update = gr.update(value=status_text)
        table_update = gr.update(value=data)
        latest_output_update = gr.update(value=latest_output, visible=bool(latest_output)) if latest_output else gr.update()
        hist1_update = gr.update(value=hist1, visible=bool(hist1))
        hist2_update = gr.update(value=hist2, visible=bool(hist2))
        hist3_update = gr.update(value=hist3, visible=bool(hist3))

        return (queue_update, table_update, latest_output_update, hist1_update, hist2_update, hist3_update)

def cancel_task(task_id):
    """取消指定任务"""
    with queue_lock:
        if task_id in queue_status and queue_status[task_id]['status'] == TaskStatus.PENDING:
            queue_status[task_id]['status'] = TaskStatus.CANCELLED
            return f"任务 {task_id[:8]}... 已取消"
    return "无法取消该任务"

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
        return {
            segments_preview: gr.update(value=data, visible=True, type="array"),
        }
    else:
        df = pd.DataFrame([], columns=[i18n("序号"), i18n("分句内容"), i18n("Token数")])
        return {
            segments_preview: gr.update(value=df),
        }

def on_method_select(emo_control_method):
    if emo_control_method == 1:
        return (gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
                )
    elif emo_control_method == 2:
        return (gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False)
                )
    elif emo_control_method == 3:
        return (gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True)
                )
    else:
        return (gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
                )

def update_timbre_preview(selected_path):
    """根据下拉选中的路径，更新试听播放器"""
    if selected_path and os.path.exists(selected_path):
        return gr.update(value=selected_path, visible=True)
    return gr.update(value=None, visible=False)

def refresh_timbres():
    """刷新 saved_timbres 列表，并同时更新下拉与试听"""
    choices = list_timbres()
    value = get_default_timbre()  # 使用新的默认音色获取函数
    dropdown_update = gr.update(choices=choices, value=value)
    preview_update = gr.update(value=value, visible=bool(value))
    return dropdown_update, preview_update

def auto_refresh_queue():
    """自动刷新队列状态"""
    return get_queue_status()

with gr.Blocks(title="IndexTTS Demo") as demo:
    mutex = threading.Lock()
    gr.HTML('''
    <h2><center>IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech</h2>
<p align="center">
<a href='https://arxiv.org/abs/2506.21619'><img src='https://img.shields.io/badge/ArXiv-2506.21619-red'></a>
</p>
    ''')
    with gr.Tab(i18n("音频生成")):
        # 队列状态显示区域
        with gr.Row():
            with gr.Column(scale=2):
                queue_status_display = gr.Markdown(value="### 队列状态\n- 等待初始化...")
            with gr.Column(scale=1):
                refresh_queue_btn = gr.Button("🔄 刷新队列状态", variant="secondary")
                clear_queue_btn = gr.Button("🗑️ 清除已完成任务", variant="secondary")
        
        # 队列任务列表
        with gr.Row():
            queue_table = gr.Dataframe(
                headers=[i18n("序号"), i18n("文本预览"), i18n("状态"), i18n("提交时间")],
                label=i18n("任务队列（最近10个）"),
                interactive=False
            )
        
        with gr.Row():
            # 音色参考音频改为从 saved_timbres 选择
            timbre_choices = list_timbres()
            default_timbre = get_default_timbre()  # 使用新的默认音色获取函数

            with gr.Column():
                prompt_audio = gr.Dropdown(
                    label=i18n("音色参考音频（从 saved_timbres 选择）"),
                    key="prompt_audio",
                    choices=timbre_choices,
                    value=default_timbre,
                    interactive=True,
                )
                refresh_timbres_btn = gr.Button(i18n("刷新音色列表"), variant="secondary")

                # 音色试听播放器
                timbre_preview = gr.Audio(
                    label=i18n("音色试听"),
                    value=default_timbre,
                    visible=bool(default_timbre),
                    autoplay=False
                )

            with gr.Column():
                input_text_single = gr.TextArea(label=i18n("文本"), key="input_text_single",
                                                placeholder=i18n("请输入目标文本"),
                                                info=f"{i18n('当前模型版本')}{tts.model_version or '1.0'}")
                with gr.Row():
                    gen_button = gr.Button(i18n("➕ 添加到队列"), key="gen_button", interactive=True, variant="primary")
                    queue_info = gr.Textbox(label="", value="点击按钮将任务添加到生成队列", interactive=False)

        # 当前生成结果和历史记录区域
        with gr.Row():
            with gr.Column():
                gr.Markdown("### " + i18n("当前生成结果"))
                output_audio = gr.Audio(label=i18n("最新生成"), visible=True, key="output_audio")
            
            with gr.Column():
                with gr.Row():
                    gr.Markdown("### " + i18n("生成历史（最近3个）"))
                    refresh_history_btn = gr.Button(i18n("刷新历史"), size="sm", variant="secondary")
                history_audio_1 = gr.Audio(label=i18n("历史 1（最新）"), visible=False)
                history_audio_2 = gr.Audio(label=i18n("历史 2"), visible=False)
                history_audio_3 = gr.Audio(label=i18n("历史 3（最旧）"), visible=False)

        with gr.Accordion(i18n("功能设置")):
            # 情感控制选项部分
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES,
                    type="index",
                    value=EMO_CHOICES[0], label=i18n("情感控制方式"))

        # 情感参考音频部分
        with gr.Group(visible=False) as emotion_reference_group:
            with gr.Row():
                emo_upload = gr.Audio(label=i18n("上传情感参考音频"), type="filepath")

            with gr.Row():
                emo_weight = gr.Slider(label=i18n("情感权重"), minimum=0.0, maximum=1.6, value=0.8, step=0.01)

        # 情感随机采样
        with gr.Row():
            emo_random = gr.Checkbox(label=i18n("情感随机采样"), value=False, visible=False)

        # 情感向量控制部分
        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label=i18n("喜"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec2 = gr.Slider(label=i18n("怒"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec3 = gr.Slider(label=i18n("哀"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec4 = gr.Slider(label=i18n("惧"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                with gr.Column():
                    vec5 = gr.Slider(label=i18n("厌恶"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec6 = gr.Slider(label=i18n("低落"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec7 = gr.Slider(label=i18n("惊喜"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)
                    vec8 = gr.Slider(label=i18n("平静"), minimum=0.0, maximum=1.4, value=0.0, step=0.05)

        with gr.Group(visible=False) as emo_text_group:
            with gr.Row():
                emo_text = gr.Textbox(label=i18n("情感描述文本"),
                                      placeholder=i18n("请输入情绪描述（或留空以自动使用目标文本作为情绪描述）"),
                                      value="",
                                      info=i18n("例如：高兴，愤怒，悲伤等"))

        with gr.Accordion(i18n("高级生成参数设置"), open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"**{i18n('GPT2 采样设置')}** _{i18n('参数会影响音频多样性和生成速度详见')} [Generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)._")
                    with gr.Row():
                        do_sample = gr.Checkbox(label="do_sample", value=True, info=i18n("是否进行采样"))
                        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=2.0, value=0.8, step=0.1)
                    with gr.Row():
                        top_p = gr.Slider(label="top_p", minimum=0.0, maximum=1.0, value=0.8, step=0.01)
                        top_k = gr.Slider(label="top_k", minimum=0, maximum=100, value=30, step=1)
                        num_beams = gr.Slider(label="num_beams", value=3, minimum=1, maximum=10, step=1)
                    with gr.Row():
                        repetition_penalty = gr.Number(label="repetition_penalty", precision=None, value=10.0, minimum=0.1, maximum=20.0, step=0.1)
                        length_penalty = gr.Number(label="length_penalty", precision=None, value=0.0, minimum=-2.0, maximum=2.0, step=0.1)
                    max_mel_tokens = gr.Slider(label="max_mel_tokens", value=1500, minimum=50, maximum=tts.cfg.gpt.max_mel_tokens, step=10, info=i18n("生成Token最大数量，过小导致音频被截断"), key="max_mel_tokens")
                with gr.Column(scale=2):
                    gr.Markdown(f'**{i18n("分句设置")}** _{i18n("参数会影响音频质量和生成速度")}_')
                    with gr.Row():
                        initial_value = max(20, min(tts.cfg.gpt.max_text_tokens, cmd_args.gui_seg_tokens))
                        max_text_tokens_per_segment = gr.Slider(
                            label=i18n("分句最大Token数"), value=initial_value, minimum=20, maximum=tts.cfg.gpt.max_text_tokens, step=2, key="max_text_tokens_per_segment",
                            info=i18n("建议80~200之间，值越大，分句越长；值越小，分句越碎；过小过大都可能导致音频质量不高"),
                        )
                    with gr.Accordion(i18n("预览分句结果"), open=True) as segments_settings:
                        segments_preview = gr.Dataframe(
                            headers=[i18n("序号"), i18n("分句内容"), i18n("Token数")],
                            key="segments_preview",
                            wrap=True,
                        )
            advanced_params = [
                do_sample, top_p, top_k, temperature,
                length_penalty, num_beams, repetition_penalty, max_mel_tokens,
            ]

        if len(example_cases) > 0:
            gr.Examples(
                examples=example_cases,
                examples_per_page=20,
                inputs=[prompt_audio,
                        emo_control_method,
                        input_text_single,
                        emo_upload,
                        emo_weight,
                        emo_text,
                        vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
            )

        # 事件绑定
        emo_control_method.select(on_method_select,
            inputs=[emo_control_method],
            outputs=[emotion_reference_group,
                     emo_random,
                     emotion_vector_group,
                     emo_text_group]
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

        # 下拉变化 => 更新试听
        prompt_audio.change(
            update_timbre_preview,
            inputs=[prompt_audio],
            outputs=[timbre_preview]
        )

        # 刷新列表 => 同时更新下拉和试听
        refresh_timbres_btn.click(
            refresh_timbres,
            inputs=[],
            outputs=[prompt_audio, timbre_preview]
        )

        # 生成按钮 - 现在添加到队列
        gen_button.click(
            add_to_queue,
            inputs=[emo_control_method, prompt_audio, input_text_single, emo_upload, emo_weight,
                    vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                    emo_text, emo_random,
                    max_text_tokens_per_segment,
                    *advanced_params,
                    ],
            outputs=[queue_status_display, queue_table, output_audio]
        ).then(
            lambda: gr.update(value="✅ 任务已添加到队列，请等待处理..."),
            outputs=[queue_info]
        )
        
        # 刷新队列状态
        refresh_queue_btn.click(
            get_queue_status,
            inputs=[],
            outputs=[queue_status_display, queue_table, output_audio]
        )
        
        # 清除已完成任务
        clear_queue_btn.click(
            clear_completed_tasks,
            inputs=[],
            outputs=[queue_status_display, queue_table, output_audio]
        )
        
        # 刷新历史按钮 - 独立更新历史显示
        refresh_history_btn.click(
            refresh_history,
            inputs=[],
            outputs=[history_audio_1, history_audio_2, history_audio_3]
        )
        # 设置持续运行的队列状态刷新
        demo.load(
            continuous_queue_refresh,
            inputs=[],
            outputs=[queue_status_display, queue_table, output_audio],
            show_progress="hidden"  # 隐藏进度条，避免界面闪烁
        )

if __name__ == "__main__":
    # 启用队列功能，支持多个用户排队生成
    demo.queue(
        max_size=50,  # 增加队列长度以支持更多任务
        default_concurrency_limit=1  # 同时处理的请求数（设为1确保顺序处理）
    )
    
    demo.launch(
        server_name=cmd_args.host,
        server_port=cmd_args.port,
        share=False
    )