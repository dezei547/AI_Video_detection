import cv2
import os
import time
import torch
from ultralytics import YOLO
from tqdm import tqdm
import csv
import re 
import concurrent.futures
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import insightface
# from funasr import AutoModel
from rapidocr_paddle import RapidOCR
import ffmpeg
import whisper
import json
from collections import defaultdict
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
class AnalysisRequest(BaseModel):
    """API request parameters model"""
    sample_rate: Optional[int] = 1
    batch_size: Optional[int] = 16
    nsfw_threshold: Optional[float] = 0.7
    violence_threshold: Optional[float] = 0.7
    face_threshold: Optional[float] = 0.5
    scene_threshold: Optional[float] = 0.5


class EnhancedVideoContentAnalyzer:
    def __init__(self):
        """Initialize analyzer, setup directory structure and load models"""
        # Initialize directory structure
        self.OUTPUT_DIRS = {
            'faces': 'detected_faces',
            'gallery': 'face_gallery',
            'scenes': 'scene_annotations',
            'nsfw': 'nsfw_detections',
            'violence': 'violence_detections',
            'smoking': 'smoking_detections', 
            'abnormalities': 'abnormalities',  # 新增异常检测目录
            'logs': 'logs'
        }
                #初始化OCR模型，使用rapidocr_paddle
        self.ocr = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)

        # 敏感词库和敏感人脸库
        self.SENSITIVE_WORDS_FILE = 'sensitive_words.txt'
        self.FACE_DB_DIR = 'face_database'  # 人脸库目录


        self.sensitive_words = self._load_sensitive_words()
        self._setup_dirs()
        self.BLACK_THRESHOLD = 0.1  # 黑场像素强度阈值(0-1)
        self.FROZEN_THRESHOLD = 125  # 静帧持续帧数阈值
        self.BLACK_DURATION_THRESHOLD = 50  # 黑场持续帧数阈值     
        
        # self.asr_model = AutoModel(
        #     model='models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        #     model_revision="v2.0.4",
        #     vad_model='models/speech_fsmn_vad_zh-cn-16k-common-pytorch' ,
        #     vad_model_revision="v2.0.4",
        #     punc_model='models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
        #     punc_model_revision="v2.0.4",
        #     timestamp_model=True
        # )
        """
        初始化Whisper模型
        :param model_size: 模型大小 (tiny, base, small, medium, large)
        :param sensitive_words: 敏感词列表
        """
        self.whispermodel = whisper.load_model("tiny",download_root="models/whisper")




    
        self.FACE_SIMILARITY_THRESHOLD = 0.5  # 人脸相似度阈值   
        # Model path configuration
        self.MODEL_PATHS = {
            'face': 'models/yolov11n-face.pt',
            'nsfw': 'models/NSFW_YOLOv8.pt',
            'violence': 'models/yolov8-violence.pt',
            'smoking': 'models/yolov8_smoking.pt'
        }
        self.scene_model = None  # 将在_load_models中初始化
        self.scene_classes = []  # 场景分类标签
        model_dir = os.path.join(os.getcwd())
        os.makedirs(model_dir, exist_ok=True)
        self.face_model = insightface.app.FaceAnalysis(
            name='buffalo_l',  # 高精度模型
            root=model_dir,
            providers=['CUDAExecutionProvider'],  # 使用GPU
            download=False
        )
        self.face_model.prepare(ctx_id=0, det_size=(640, 640))
        # Load models
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models = self._load_models()
        self.FACE_DB = self._load_face_db()  # 加载人脸库
        print(f"Using device: {self.device}")
        
        # Class mappings
        self.NSFW_CLASSES = {
            'human_porn': '真人色情',
            'human_porns': '真人色情(多人物)',
            'anime_itchi': '动漫情色',
            'anime_hentai': '动漫色情',
            'human_nude': '真人裸露'
        }
        
        self.VIOLENCE_CLASSES = {
            'NonViolence': '无暴力',
            'Violence': '暴力行为'
        }
        
        self.NSFW_LABELS = list(self.NSFW_CLASSES.keys())
        self.VIOLENCE_LABELS = list(self.VIOLENCE_CLASSES.keys())
        
        # Store last results for API
        self.last_results = None

    def _create_segmented_results(self, results: Dict[str, Any], segment_duration: int = 5) -> Dict[str, Any]:
        """
        将检测结果按时间分段汇总
        
        Args:
            results: 原始检测结果
            segment_duration: 分段时长(秒)
            
        Returns:
            分段汇总结果（包含所有分段）
        """
        fps = results['video_info']['fps']
        total_duration = results['video_info']['duration']
        total_segments = int(total_duration // segment_duration) + 1
        
        # 初始化分段数据结构 - 创建所有分段
        segments = {}
        for seg_idx in range(total_segments):
            start_time = seg_idx * segment_duration
            end_time = min((seg_idx + 1) * segment_duration, total_duration)
            segments[f"{start_time}-{end_time}s"] = {
                "segment": f"{start_time}-{end_time}s",
                "adult_content": False,
                "smoking": False,
                "violence": False,
                "black_scenes": False,
                "frozen_scenes": False,
                "matched_faces": [],
                "sensitive_words": [],
                "all_ocr_texts": [],
                'video_text':[]
            }
        # 处理语音识别全部
        videotext_segments = results.get('video_text', {})
        for segment_key, video_texts in videotext_segments.items():
            if segment_key in segments:
                
                segments[segment_key]['video_text'] = video_texts
            else:
                pass
            

        for segment_key in segments:
            if 'all_ocr_texts' not in segments[segment_key]:
                segments[segment_key]['video_text'] = []          
        # 处理敏感词检测结果

        for word_item in results.get('sensitive_words', []):
           
            frame_num = word_item['frame']
            timestamp = frame_num / fps
            segment_idx = int(timestamp // segment_duration)
            start_time = segment_idx * segment_duration
            end_time = min((segment_idx + 1) * segment_duration, total_duration)
            segment_key = f"{start_time}-{end_time}s"
            
            if segment_key in segments:
                # 去重添加敏感词
                for word in word_item['words']:
                    print(f"Sensitive word detected: {word}")
                    if word not in segments[segment_key]['sensitive_words']:
                        segments[segment_key]['sensitive_words'].append(word)
        
        # 处理成人内容检测
        for nsfw_item in results.get('nsfw', []):
            if nsfw_item['is_detected']:
                # 根据帧数和帧率计算时间戳
                frame_num = nsfw_item['frame']
                timestamp = frame_num / fps
                segment_idx = int(timestamp // segment_duration)
                start_time = segment_idx * segment_duration
                end_time = min((segment_idx + 1) * segment_duration, total_duration)
                segment_key = f"{start_time}-{end_time}s"
                
                if segment_key in segments:
                    segments[segment_key]['adult_content'] = True
        
        # 处理吸烟检测 (使用专门的抽烟检测模型)
        for smoking_item in results.get('smoking', []):
            if smoking_item['is_detected']:
                # 根据帧数和帧率计算时间戳
                frame_num = smoking_item['frame']
                timestamp = frame_num / fps
                segment_idx = int(timestamp // segment_duration)
                start_time = segment_idx * segment_duration
                end_time = min((segment_idx + 1) * segment_duration, total_duration)
                segment_key = f"{start_time}-{end_time}s"
                
                if segment_key in segments:
                    segments[segment_key]['smoking'] = True
        
        # 处理暴力检测
        for violence_item in results.get('violence', []):
            if violence_item['is_detected']:
                # 根据帧数和帧率计算时间戳
                frame_num = violence_item['frame']
                timestamp = frame_num / fps
                segment_idx = int(timestamp // segment_duration)
                start_time = segment_idx * segment_duration
                end_time = min((segment_idx + 1) * segment_duration, total_duration)
                segment_key = f"{start_time}-{end_time}s"
                
                if segment_key in segments:
                    segments[segment_key]['violence'] = True
        
        # 处理黑场检测
        for black_item in results.get('abnormalities', {}).get('black_frames', []):
            # 根据帧数和帧率计算时间戳
            frame_num = black_item['frame']
            timestamp = frame_num / fps
            segment_idx = int(timestamp // segment_duration)
            start_time = segment_idx * segment_duration
            end_time = min((segment_idx + 1) * segment_duration, total_duration)
            segment_key = f"{start_time}-{end_time}s"
            
            if segment_key in segments:
                segments[segment_key]['black_scenes'] = True
        
        # 处理静帧检测
        for frozen_item in results.get('abnormalities', {}).get('frozen_frames', []):
            # 根据帧数和帧率计算时间戳
            frame_num = frozen_item['frame']
            timestamp = frame_num / fps
            segment_idx = int(timestamp // segment_duration)
            start_time = segment_idx * segment_duration
            end_time = min((segment_idx + 1) * segment_duration, total_duration)
            segment_key = f"{start_time}-{end_time}s"
            
            if segment_key in segments:
                segments[segment_key]['frozen_scenes'] = True
        
        # 处理人脸匹配结果
        for face_item in results.get('faces', []):
            for face in face_item.get('faces', []):
                matched_person = face.get('matched_person')
                if matched_person and matched_person != '未知':
                    # 根据帧数和帧率计算时间戳
                    frame_num = face_item['frame']
                    timestamp = frame_num / fps
                    segment_idx = int(timestamp // segment_duration)
                    start_time = segment_idx * segment_duration
                    end_time = min((segment_idx + 1) * segment_duration, total_duration)
                    segment_key = f"{start_time}-{end_time}s"
                    
                    if segment_key in segments and matched_person not in segments[segment_key]['matched_faces']:
                        segments[segment_key]['matched_faces'].append(matched_person)
        
        # 处理OCR敏感文本
        for ocr_item in results.get('ocr', []):
            if ocr_item.get('text'):
                # OCR结果已经有frame_num，直接计算时间戳
                frame_num = ocr_item['frame_num']
                timestamp = frame_num / fps
                segment_idx = int(timestamp // segment_duration)
                start_time = segment_idx * segment_duration
                end_time = min((segment_idx + 1) * segment_duration, total_duration)
                segment_key = f"{start_time}-{end_time}s"
                
                if segment_key in segments:
                    # 去重添加OCR文本
                    for text in ocr_item['text']:
                        if text not in segments[segment_key]['sensitive_words']:
                            segments[segment_key]['sensitive_words'].append(text)
        # 处理场景检测结果
        for scene_item in results.get('scenes', []):
            if scene_item['objects']:
                frame_num = scene_item['frame']
                timestamp = frame_num / fps
                segment_idx = int(timestamp // segment_duration)
                start_time = segment_idx * segment_duration
                end_time = min((segment_idx + 1) * segment_duration, total_duration)
                segment_key = f"{start_time}-{end_time}s"
                
                if segment_key in segments:
                    # 确保 scene_objects 字段存在
                    if 'scene' not in segments[segment_key]:
                        segments[segment_key]['scene'] = []
                    
                    # 提取场景中的物体信息
                    for obj in scene_item['objects']:
                        class_name = obj['chinese_name']
                        # 去重添加场景物体类别名称
                        if class_name not in segments[segment_key]['scene']:
                            segments[segment_key]['scene'].append(class_name)     
        # 处理OCR全部文本
        all_ocr_segments = results.get('all_ocr_segments', {})
        for segment_key, ocr_texts in all_ocr_segments.items():
            if segment_key in segments:
                # 直接使用OCR分段结果
                segments[segment_key]['all_ocr_texts'] = ocr_texts
            else:
                # 如果分段键不匹配，尝试找到对应的时间段
                # 例如：all_ocr_segments中的'0-5s'对应segments中的'0-5s'
                # 这里确保键名一致
                pass
            
        # 确保所有分段都有all_ocr_texts，即使为空
        for segment_key in segments:
            if 'all_ocr_texts' not in segments[segment_key]:
                segments[segment_key]['all_ocr_texts'] = []        
        # 转换为列表格式 - 保留所有分段
        segmented_list = []
        for seg_idx in range(total_segments):
            start_time = seg_idx * segment_duration
            end_time = min((seg_idx + 1) * segment_duration, total_duration)
            segment_key = f"{start_time}-{end_time}s"
            
            if segment_key in segments:
                segmented_list.append(segments[segment_key])
            else:
                # 如果分段不存在，创建空分段
                segmented_list.append({
                    "segment": segment_key,
                    "sensitive_words": [],
                    "adult_content": False,
                    "smoking": False,
                    "violence": False,
                    "black_scenes": False,
                    "frozen_scenes": False,
                    "matched_faces": [],
                    "all_ocr_texts": [],
                    "video_text": [],
                    "scenes": []  # 新增空场景对象列表
                })
        
        return {
            "video_info": results['video_info'],
            "analysis_params": results['analysis_params'],
            "segments": segmented_list,
            "total_segments": len(segmented_list)
        }
    #加载人脸库 
    def _load_face_db(self, paths=None):
        """Load face databases from multiple directories"""
        face_db = {}
        paths = paths or [self.FACE_DB_DIR]  # Default to original path if none provided
        
        for db_path in paths:
            if os.path.exists(db_path):
                for person_name in os.listdir(db_path):
                    person_dir = os.path.join(db_path, person_name)
                    if os.path.isdir(person_dir):
                        embeddings = []
                        for img_file in os.listdir(person_dir):
                            img_path = os.path.join(person_dir, img_file)
                            img = cv2.imread(img_path)
                            if img is not None:
                                faces = self.face_model.get(img)
                                if faces:
                                    embeddings.append(faces[0].embedding)
                        if embeddings:
                            # Handle case where same person exists in multiple DBs
                            if person_name in face_db:
                                face_db[person_name] = np.concatenate((face_db[person_name], np.mean(embeddings, axis=0)))
                            else:
                                face_db[person_name] = np.mean(embeddings, axis=0)
        
        return face_db
    def _compare_with_face_db(self, frame):
        """人脸比对逻辑"""

        faces = self.face_model.get(frame)
        
        if not faces or not self.FACE_DB:
            return None, 0

        # 提取待比对特征
        query_embedding = faces[0].embedding
        matched_person, max_similarity = None, 0

        # 计算余弦相似度
        for name, db_embedding in self.FACE_DB.items():
            sim = np.dot(query_embedding, db_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding))
            if sim > max_similarity:
                max_similarity = sim
                matched_person = name
            
        return (matched_person, max_similarity) if max_similarity > self.FACE_SIMILARITY_THRESHOLD else (None, 0)

    ##########音频检测#########
    def _timestamp_to_frame(self, timestamp: str, fps: float) -> int:
        """
        将时间戳字符串转换为帧数
        
        Args:
            timestamp: 时间戳字符串 (HH:MM:SS.ms)
            fps: 视频帧率
            
        Returns:
            int: 对应的帧数
        """
        try:
            # 分割小时、分钟、秒和毫秒
            hh_mm_ss, ms = timestamp.split('.') if '.' in timestamp else (timestamp, '000')
            hh, mm, ss = hh_mm_ss.split(':')
            
            # 计算总秒数
            total_seconds = (int(hh) * 3600 + 
                            int(mm) * 60 + 
                            int(ss) + 
                            int(ms) / 1000)
            
            # 计算帧数
            return int(total_seconds * fps)
        except Exception as e:
            print(f"时间戳转换错误: {timestamp} - {str(e)}")
            return 0
    def _load_sensitive_words(self, paths=None):
        """Load sensitive words from multiple files, supporting various formats:
        - One word per line
        - Words separated by Chinese comma (，)
        - Words separated by English comma (,)
        - Words separated by Chinese顿号 (、)
        """
        sensitive_words = []
        paths = paths or [self.SENSITIVE_WORDS_FILE]  # Default to original path if none provided
        
        for path in paths:
            if not os.path.exists(path):
                continue
                
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Handle different formats
                    if '\n' in content:  # Line-separated format
                        words = [line.strip() for line in content.split('\n') if line.strip()]
                    elif '，' in content:  # Chinese comma separated
                        words = [word.strip() for word in content.split('，') if word.strip()]
                    elif ',' in content:  # English comma separated
                        words = [word.strip() for word in content.split(',') if word.strip()]
                    elif '、' in content:  # Chinese顿号 separated
                        words = [word.strip() for word in content.split('、') if word.strip()]
                    else:  # Single word or unknown format
                        words = [content.strip()] if content.strip() else []
                        
                    sensitive_words.extend(words)
                    
            except Exception as e:
                print(f"Warning: Failed to load sensitive words from {path}: {str(e)}")
                continue
                
        return list(set(sensitive_words))  # Remove duplicates
    #查询敏感词时间戳
    def get_phrase_timestamps(self, segment, target_phrase):
        """在单个段落中查找特定短语的精确时间戳"""
        phrase_words = target_phrase.replace(" ", "")  # 中文通常不需要空格
        phrase_length = len(phrase_words)
        phrase_timestamps = []
        
        words = segment.get("words", [])
        for i in range(len(words) - phrase_length + 1):
            # 组合连续的字/词来匹配目标短语
            combined_word = "".join([words[i+j]["word"] for j in range(phrase_length)])
            
            if combined_word == target_phrase:
                start_time = words[i]["start"]
                end_time = words[i + phrase_length - 1]["end"]
                phrase_timestamps.append({
                    "start": start_time,
                    "end": end_time
                })
        
        return phrase_timestamps
    
    def find_sensitive_words(self, segments):
        """在识别结果中查找敏感词，支持完整短语匹配"""
        sensitive_results = []
        
        if not self.sensitive_words:
            return sensitive_results
        
        for segment in segments:
            segment_text = segment["text"]
            segment_start = segment["start"]
            segment_end = segment["end"]
            
            for sensitive_word in self.sensitive_words:
                # 先检查整个段落中是否包含敏感词（快速筛选）
                if sensitive_word not in segment_text:
                    continue
                    
                # 查找精确的时间戳位置
                phrase_matches = self.get_phrase_timestamps(segment, sensitive_word)
                
                if phrase_matches:
                    for match in phrase_matches:
                        sensitive_results.append({
                            "exact_timestamp": f"{self.format_timestamps(match['start'])}-{self.format_timestamps(match['end'])}",
                            "sensitive_word": sensitive_word,
                            "segment_text": segment_text,
                            "segment_timestamp": f"{self.format_timestamps(segment_start)}-{self.format_timestamps(segment_end)}"
                        })
                else:
                    # 如果没有找到完整短语匹配，回退到单词级搜索
                    for word_info in segment.get("words", []):
                        if sensitive_word in word_info["word"]:
                            sensitive_results.append({
                                "exact_timestamp": f"{self.format_timestamps(word_info['start'])}-{self.format_timestamps(word_info['end'])}",
                                "sensitive_word": sensitive_word,
                                "segment_text": segment_text,
                                "segment_timestamp": f"{self.format_timestamps(segment_start)}-{self.format_timestamps(segment_end)}"
                            })
        
        return sensitive_results

    def extract_audio(self, video_path, audio_path):
        """使用FFmpeg提取音频并转换为16kHz单声道"""
        try:
            (
                ffmpeg.input(video_path)
                .output(audio_path, ac=1, ar=16000, loglevel="quiet")
                .overwrite_output()
                .run()
            )
            return True
        except Exception as e:
            print(f"音频提取失败: {str(e)}")
            return False


    def format_timestamps(self, seconds):
        """将秒数转换为HH:MM:SS.ms格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    
    def transcribe_with_timestamps(self, audio_path):
        """执行带时间戳的语音识别"""
        try:
            # 使用Whisper进行推理，启用时间戳
            result = self.whispermodel.transcribe(
                audio_path,
                language="zh",
                verbose=False,
                initial_prompt="简体中文",
                word_timestamps=True  # 获取单词级时间戳
            )
            return result
        except Exception as e:
            print(f"语音识别失败: {str(e)}")
            return None
    def expand_to_frame_records(self,sensitive_results, fps=25, sample_interval=1):
        """
        将敏感词时间戳扩展为帧级结构化记录（支持采样间隔）
        
        参数:
            sensitive_results: 原始敏感词检测结果列表
            fps: 视频帧率（默认为25帧/秒）
            sample_interval: 采样间隔帧数（默认为1，即每帧都保存）
            
        返回:
            按帧扩展后的结构化结果列表（按采样间隔）
        """
        frame_records = []
        
        for item in sensitive_results:
            # 解析原始时间戳
            start_time_str, end_time_str = item['exact_timestamp'].split('-')
            text = item['segment_text']
            
            # 转换时间格式为秒数
            start_sec = sum(x * float(t) for x, t in zip([3600, 60, 1], start_time_str.split(":")))
            end_sec = sum(x * float(t) for x, t in zip([3600, 60, 1], end_time_str.split(":")))
            
            # 计算起止帧号
            start_frame = int(start_sec * fps)
            end_frame = int(end_sec * fps)
            
            # 提取敏感词（去重处理）
            found_words = list(set(item['sensitive_word'].split('|'))) if '|' in item['sensitive_word'] else [item['sensitive_word']]
            
            # 按采样间隔生成记录
            for frame in range(start_frame, end_frame + 1, sample_interval):
                frame_records.append({
                    'frame': frame,
                    'timestamp': round(frame / fps, 3),  # 保留3位小数
                    'start_time': start_time_str,
                    'end_time': end_time_str,
                    'text': text.strip(),
                    'words': found_words,
                    'chinese_words': found_words,
                    'frame_range': f"{start_frame}-{end_frame}",
                    'sample_rate': sample_interval
                })
        print(frame_records)
        return frame_records    

    def process_audio(self, video_path, output_file=None):
        """完整处理流程"""
        # 临时音频文件
        temp_audio = "temp_audio.wav"
        
        # 1. 提取音频
        if not self.extract_audio(video_path, temp_audio):
            return False, None
        
        # 2. 语音识别
        result = self.transcribe_with_timestamps(temp_audio)
        
        if not result:
            print("识别失败")
            return False, None
        
        # 3. 查找敏感词
        sensitive_results = self.find_sensitive_words(result["segments"])    
          
        # 4. 创建按5秒分段的音频识别文本
        segmented_audio_texts = self._create_audio_segments(result["segments"])       
        # 清理临时文件
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        
        return sensitive_results, segmented_audio_texts
    def _create_audio_segments(self, segments, segment_duration=5):
        """
        将音频识别结果按指定时长分段
        
        Args:
            segments: 语音识别的分段结果
            segment_duration: 分段时长，默认5秒
            
        Returns:
            按时间段分组的音频文本字典
        """
        segmented_texts = {}
        
        for segment in segments:
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text']
            
            # 如果分段有words信息，使用更精确的单词级时间戳
            if 'words' in segment and segment['words']:
                # 使用单词级时间戳进行更精确的分段
                for word_info in segment['words']:
                    word_start = word_info['start']
                    word_end = word_info['end']
                    word_text = word_info['word']
                    
                    # 计算单词属于哪个分段
                    segment_start = (int(word_start) // segment_duration) * segment_duration
                    segment_end = segment_start + segment_duration
                    segment_key = f"{segment_start}-{segment_end}s"
                    
                    # 确保分段存在
                    if segment_key not in segmented_texts:
                        segmented_texts[segment_key] = []
                    
                    # 添加单词到对应分段
                    if word_text not in segmented_texts[segment_key]:
                        segmented_texts[segment_key].append(word_text)
            else:
                # 如果没有单词级时间戳，使用段落级时间戳
                # 计算该段落跨越的所有分段
                current_segment_start = (int(start_time) // segment_duration) * segment_duration
                end_segment_start = (int(end_time) // segment_duration) * segment_duration
                
                # 处理段落跨越的每个分段
                segment_start = current_segment_start
                while segment_start <= end_segment_start:
                    segment_end = segment_start + segment_duration
                    segment_key = f"{segment_start}-{segment_end}s"
                    
                    # 确保分段存在
                    if segment_key not in segmented_texts:
                        segmented_texts[segment_key] = []
                    
                    # 添加完整文本到分段（去重）
                    if text and text not in segmented_texts[segment_key]:
                        segmented_texts[segment_key].append(text)
                    
                    segment_start += segment_duration
        
        # 将每个分段中的单词列表合并成文本
        for segment_key, word_list in segmented_texts.items():
            if word_list:
                # 将单词列表合并成连续的文本
                segmented_texts[segment_key] = [''.join(word_list)]
            else:
                segmented_texts[segment_key] = []
        
        return segmented_texts
    ##########音频检测#########

    ################OCR识别######################################################
################OCR识别######################################################
    def _detect_sensitive_in_ocr(self, ocr_results: List[Dict[str, Any]], fps: float) -> List[Dict[str, Any]]:
        """检测OCR结果中的敏感词"""
        sensitive_ocr = []
        
        for frame in ocr_results:
            if not frame['text']:
                continue
                
            frame_num = frame['frame_num']
            timestamp = frame_num / fps
            
            for text in frame['text']:
                found_words = []
                for word in self.sensitive_words:
                    if word in text:
                        found_words.append(word)
                
                if found_words:
                    sensitive_ocr.append({
                        'frame': frame_num,
                        'timestamp': timestamp,
                        'text': text,
                        'words': found_words,
                        'chinese_words': found_words,
                        'frame_range': f"{frame_num}-{frame_num}",  # OCR检测单帧范围
                        'sample_rate': 1,
                        'source': 'ocr'  # 标记来源为OCR
                    })
        
        return sensitive_ocr

    def process_batch(self, frames: List[np.ndarray], frame_numbers: List[int], fps: float) -> List[Optional[List[Dict[str, Any]]]]:
        """批量处理帧，返回所有OCR结果（包括敏感词和非敏感词）"""
        results = []
        for frame, frame_num in zip(frames, frame_numbers):
            try:
                # 分辨率缩放
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                ocr_result = self.ocr(small_frame)
                
                if not ocr_result or not ocr_result[0]:
                    results.append(None)
                    continue
                    
                frame_results = []
                for box, text, score in ocr_result[0]:
                    scaled_box = [[int(x*2) for x in pt] for pt in box]
                    frame_results.append({
                        'text': text,
                        'score': float(score),
                        'bbox': scaled_box,
                        'frame_num': frame_num,
                        'timestamp': frame_num / fps,
                        'has_sensitive': any(word in text for word in self.sensitive_words)  # 标记是否包含敏感词
                    })
                
                results.append(frame_results if frame_results else None)
            except Exception as e:
                print(f"处理帧 {frame_num} 时出错: {str(e)}")
                results.append(None)
        return results

    def process_video(self, video_path: str, batch_size: int = 8, frame_interval: int = 10) -> Dict[str, Any]:
        """处理视频文件，返回敏感词OCR结果和所有OCR的分段结果"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        print(f"开始处理视频: {video_path} (总帧数: {total_frames}, FPS: {fps:.1f}, 时长: {duration:.2f}s)")

        # 存储所有OCR结果
        all_ocr_dict = {}
        # 存储敏感词OCR结果
        sensitive_ocr_dict = {}
        
        frame_batch = []
        frame_num_batch = []
        frame_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                if frame_batch:
                    batch_results = self.process_batch(frame_batch, frame_num_batch, fps)
                    self._merge_all_results(batch_results, frame_num_batch, all_ocr_dict, sensitive_ocr_dict)
                break

            frame_counter += 1
            
            if frame_counter % frame_interval != 0:
                continue
                
            frame_batch.append(frame.copy())
            frame_num_batch.append(frame_counter)
            
            if len(frame_batch) == batch_size:
                batch_results = self.process_batch(frame_batch, frame_num_batch, fps)
                self._merge_all_results(batch_results, frame_num_batch, all_ocr_dict, sensitive_ocr_dict)
                
                frame_batch = []
                frame_num_batch = []
                
                sensitive_texts = sum(len(texts) for texts in sensitive_ocr_dict.values())
                all_texts = sum(len(texts) for texts in all_ocr_dict.values())
                print(f"已处理 {frame_counter}/{total_frames} 帧, 敏感文本: {sensitive_texts} 处, 总文本: {all_texts} 处")

        cap.release()
        
        # 转换为最终输出格式
        sensitive_results = [
            {'frame_num': frame_num, 'text': texts}
            for frame_num, texts in sorted(sensitive_ocr_dict.items())
        ]
        
        all_results = [
            {'frame_num': frame_num, 'text': texts}
            for frame_num, texts in sorted(all_ocr_dict.items())
        ]
        
        # 创建按时间分段的所有OCR结果
        segmented_ocr = self._create_ocr_segments(all_results, fps)
        print('分段OCR结果：',segmented_ocr)
        return {
            'sensitive_ocr': sensitive_results,  # 只包含敏感词的OCR结果
            'all_ocr_segments': segmented_ocr,   # 所有OCR结果按时间分段
            'video_info': {
                'fps': fps,
                'total_frames': total_frames,
                'duration': duration
            }
        }

    def _merge_all_results(self, batch_results: List[List[Dict]], frame_numbers: List[int], 
                        all_ocr_dict: Dict, sensitive_ocr_dict: Dict) -> None:
        """合并批量结果到两个字典：所有OCR结果和敏感词OCR结果"""
        for frame_result, frame_num in zip(batch_results, frame_numbers):
            if frame_result is None:
                continue
                
            if frame_num not in all_ocr_dict:
                all_ocr_dict[frame_num] = []
            if frame_num not in sensitive_ocr_dict:
                sensitive_ocr_dict[frame_num] = []
            
            # 提取当前帧所有文本
            all_texts = [res['text'] for res in frame_result]
            all_ocr_dict[frame_num].extend(all_texts)
            
            # 提取当前帧包含敏感词的文本
            sensitive_texts = [res['text'] for res in frame_result if res['has_sensitive']]
            sensitive_ocr_dict[frame_num].extend(sensitive_texts)

    def _create_ocr_segments(self, ocr_results: List[Dict[str, Any]], fps: float, segment_duration: int = 5) -> Dict[str, List[str]]:
        """
        将OCR结果按时间分段
        
        Args:
            ocr_results: OCR结果列表
            fps: 视频帧率
            segment_duration: 分段时长，默认5秒
            
        Returns:
            按时间段分组的OCR文本字典
        """
        segmented_texts = {}
        
        for ocr_frame in ocr_results:
            frame_num = ocr_frame['frame_num']
            timestamp = frame_num / fps
            
            # 计算该帧属于哪个分段
            segment_start = (int(timestamp) // segment_duration) * segment_duration
            segment_end = segment_start + segment_duration
            segment_key = f"{segment_start}-{segment_end}s"
            
            # 确保分段存在
            if segment_key not in segmented_texts:
                segmented_texts[segment_key] = []
            
            # 添加该帧的所有OCR文本到对应分段（去重）
            for text in ocr_frame['text']:
                if text and text not in segmented_texts[segment_key]:
                    segmented_texts[segment_key].append(text)
        
        return segmented_texts

################OCR识别######################################################          
################OCR识别######################################################


    def _setup_dirs(self) -> None:
        """Create all output directories"""
        for dir_path in self.OUTPUT_DIRS.values():
            os.makedirs(dir_path, exist_ok=True)
    def _is_black_frame(self, frame: np.ndarray) -> bool:
        """检测黑场帧"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray) < 255 * self.BLACK_THRESHOLD

    def _is_frozen_frame(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> bool:
        """检测静帧"""
        if prev_frame is None:
            return False
        # 使用MSE(均方误差)计算相似度
        mse = np.mean((prev_frame - curr_frame) ** 2)
        return mse < 10  # 阈值可根据实际情况调整
    def _load_models(self) -> Dict[str, YOLO]:
        """Load all YOLO models"""
        print("Loading models...")
        
        models = {}
        try:
            # Use Automatic Mixed Precision (AMP) for faster inference
            models['face'] = YOLO(self.MODEL_PATHS['face']).to(self.device)
            
            models['nsfw'] = YOLO(self.MODEL_PATHS['nsfw']).to(self.device)
            models['violence'] = YOLO(self.MODEL_PATHS['violence']).to(self.device)
            models['smoking'] = YOLO(self.MODEL_PATHS['smoking']).to(self.device)  # 新增抽烟模型

            # 加载ResNet场景检测模型
            self._load_scene_model()
            # Warm up models
            print("Warming up models...")
            dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
            for model in models.values():
                if hasattr(model, 'model'):
                    model.model(dummy_input) if self.device == 'cuda' else model.model(dummy_input)
            
            print("Models loaded and warmed up!")
            return models
        except Exception as e:
            raise ValueError(f"Model loading failed: {str(e)}")
    def _load_scene_model(self):
        """Load ResNet50-places365 model for scene classification"""
        try:
            
            # 模型架构
            arch = 'resnet18'
            model_file = f'models/{arch}_places365.pth.tar'
            
            # 确保模型文件存在
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Scene model not found: {model_file}")
            
            # 加载模型
            self.scene_model = models.__dict__[arch](num_classes=365)
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
            self.scene_model.load_state_dict(state_dict)
            self.scene_model.eval()
            self.scene_model.to(self.device)
            
            # 图像预处理
            self.scene_transform = trn.Compose([
                trn.Resize((256, 256)),
                trn.CenterCrop(224),
                trn.ToTensor(),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            # 加载类别标签
            classes_file = 'models/categories_places365.txt'
            if not os.path.exists(classes_file):
                raise FileNotFoundError(f"Scene classes file not found: {classes_file}")
            
            self.scene_classes = []
            with open(classes_file, 'r', encoding='utf-8') as f:
                for line in f:
                    self.scene_classes.append(line.strip().split(' ')[0][3:])
            
            print(f"Scene model loaded: {arch} with {len(self.scene_classes)} classes")
            
        except Exception as e:
            print(f"Failed to load scene model: {str(e)}")
            self.scene_model = None
    def analyze_video(
        self, 
        video_path: str, 
        sample_rate: int = 1, 
        batch_size: int = 24, 
        nsfw_threshold: float = 0.5,
        violence_threshold: float = 0.5,
        face_threshold: float = 0.5,
        scene_threshold: float = 0.5,
        frozen_threshold: int = 125,
        black_duration_threshold: int = 50,
        sensitive_words_path: Optional[List[str]] = None,  # Now accepts list of paths
        sensitive_faces_path: Optional[List[str]] = None,  # Now accepts list of paths
        whisper_model: str = "tiny"
    ) -> Dict[str, Any]:
        """
        Enhanced main analysis function with face and scene detection
        
        Args:
            video_path: Path to video file
            sample_rate: Frame sampling rate (1=process every frame, 2=every 2nd frame, etc.)
            batch_size: Batch processing size
            nsfw_threshold: NSFW detection confidence threshold
            violence_threshold: Violence detection confidence threshold
            face_threshold: Face detection confidence threshold
            scene_threshold: Scene detection confidence threshold
            frozen_threshold: 静帧检测阈值
            black_duration_threshold: 黑场持续时间阈值
            sensitive_words_path: 敏感词库文件路径
            sensitive_faces_path: 敏感人脸库目录路径
        Returns:
            Dictionary containing analysis results
        """
        if whisper_model != "tiny":
            self.whispermodel = whisper.load_model(whisper_model,download_root="models/whisper")
        if sensitive_words_path:
            self.sensitive_words = self._load_sensitive_words(sensitive_words_path)
            
        # 更新人脸库路径
        if sensitive_faces_path:
            self.FACE_DB = self._load_face_db(sensitive_faces_path)
           
        
        self.FROZEN_THRESHOLD = frozen_threshold
        self.BLACK_DURATION_THRESHOLD = black_duration_threshold
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nVideo analysis started - Resolution: {width}x{height}, Duration: {duration:.2f}s, Frames: {total_frames}, FPS: {fps}")
        print(f"Analysis params - Sample rate: {sample_rate}, Batch size: {batch_size}")
        print(f"Detection thresholds - NSFW: {nsfw_threshold}, Violence: {violence_threshold}, Face: {face_threshold}, Scene: {scene_threshold},frozenframe:{frozen_threshold},blackframe:{black_duration_threshold}" )


        # Initialize result storage
        results = {
            'video_info': {
                'path': video_path,
                'duration': duration,
                'fps': fps,
                'resolution': f"{width}x{height}",
                'total_frames': total_frames
            },
            'analysis_params': {
                'sample_rate': sample_rate,
                'batch_size': batch_size,
                'nsfw_threshold': nsfw_threshold,
                'violence_threshold': violence_threshold,
                'face_threshold': face_threshold,
                'scene_threshold': scene_threshold
            },
            'abnormalities': {
                'black_scenes': [],  # 黑场异常片段（汇总信息）
                'black_frames': [],  # 新增：存储每一帧黑场记录
                'frozen_scenes': [],  # 静帧异常片段
                'frozen_frames': []
            },
            'faces': [],
            'scenes': [],
            'nsfw': [],
            'violence': [],
            'smoking': [],# 新增抽烟检测结果
            'ocr': [],  # 新增OCR敏感词识别结果
            'all_ocr_segments': [],# 新增OCR全部识别结果
            'stats': {
                'total_faces': 0,
                'total_scenes': 0,
                'total_nsfw': 0,
                'total_violence': 0,
                'total_sensitive_words': 0,  # 新增敏感词统计
                'total_smoking': 0,
                'processed_frames': 0,
                'processing_time': 0,
                'processing_fps': 0,
                'black_scenes': 0,  # 黑场异常计数
                'frozen_scenes': 0,  # 静帧异常计数
                'total_ocr_texts': 0
            },
            'sensitive_words': [], 

        }
        
        start_time = time.time()
        frames_batch = []
        frame_indices = []
        pbar = tqdm(total=total_frames, desc="Video analysis progress")
        prev_frame = None
        black_frame_count = 0
        frozen_frame_count = 0
        current_black_start = None
        current_frozen_start = None
        black_frames_to_save = []
        frozen_frames_to_save = []
        # Main processing loop
        for frame_count in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            pbar.update(1)
            
            # Frame sampling control
            if frame_count % sample_rate != 0:
                continue
                
            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_batch.append(frame_rgb)
            frame_indices.append(frame_count)
            results['stats']['processed_frames'] += 1
            # 黑场检测
            timestamp = frame_count / fps
            if self._is_black_frame(frame):
                if current_black_start is None:
                    current_black_start = frame_count
                black_frame_count += 1
                black_frame = {
                    'frame': frame_count,
                    'timestamp': timestamp,
                    'type': 'black',
                    'chinese_type': '黑场',
                    'confidence': 1.0,
                    'is_detected': True,
                    'segment_start_frame': current_black_start,
                    'segment_duration_frames': black_frame_count,
                }
                black_frames_to_save.append(black_frame)
                
            else:
                if black_frame_count >= self.BLACK_DURATION_THRESHOLD:
                    # 记录黑场异常
                    results['abnormalities']['black_frames'].extend(black_frames_to_save)
                    results['stats']['black_scenes'] += black_frame_count
                    
                black_frame_count = 0
                current_black_start = None
                black_frames_to_save = []
            # 静帧检测
            if self._is_frozen_frame(prev_frame, frame):
                if current_frozen_start is None:
                    current_frozen_start = frame_count - 1
                frozen_frame_count += 1
                # print("检测到静止帧",frozen_frame_count)
                frozen_frame = {
                    'frame': frame_count,
                    'timestamp': timestamp,
                    'type': 'frozen_frame',
                    'chinese_type': '静止帧',
                    'confidence': 1.0,
                    'is_detected': True,
                    'segment_start_frame': current_frozen_start,
                    'segment_duration_frames': frozen_frame_count,
                }
                frozen_frames_to_save.append(frozen_frame)
            else:
                if frozen_frame_count >= self.FROZEN_THRESHOLD:
                    # 记录静帧异常
                    # print("监测到静止帧")
                    results['abnormalities']['frozen_frames'].extend(frozen_frames_to_save)
                    results['stats']['frozen_scenes'] += frozen_frame_count
                frozen_frame_count = 0
                current_frozen_start = None
                frozen_frames_to_save = []
            prev_frame = frame.copy()            
            # Batch processing
            if len(frames_batch) == batch_size:
                # Parallel execution of all detections
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    # Face and scene detection for each frame
                    face_future = executor.submit(
                        self._batch_detect_faces, 
                        frames_batch.copy(), 
                        frame_indices.copy(), 
                        face_threshold
                    )
                    scene_future = executor.submit(
                        self._batch_detect_scenes, 
                        frames_batch.copy(), 
                        frame_indices.copy(), 
                        scene_threshold
                    )
                    nsfw_future = executor.submit(
                        self._batch_detect, 
                        frames_batch.copy(), 
                        frame_indices.copy(), 
                        'nsfw', 
                        nsfw_threshold
                    )
                    violence_future = executor.submit(
                        self._batch_detect, 
                        frames_batch.copy(), 
                        frame_indices.copy(), 
                        'violence', 
                        violence_threshold
                    )
                    smoking_future = executor.submit(
                        self._batch_detect_smoking,  # 需要新增这个方法
                        frames_batch.copy(), 
                        frame_indices.copy(), 
                        scene_threshold  # 可以使用场景检测的阈值，或者单独设置
                    )                    
                    # Get results
                    face_results = face_future.result()
                    scene_results = scene_future.result()
                    nsfw_results = nsfw_future.result()
                    violence_results = violence_future.result()
                    smoking_results = smoking_future.result()
                    # Update results
                    results['faces'].extend(face_results)
                    results['scenes'].extend(scene_results)
                    results['nsfw'].extend([res for res in nsfw_results if res['is_detected']])
                    results['violence'].extend([res for res in violence_results if res['is_detected']])
                    results['smoking'].extend([res for res in smoking_results if res['is_detected']])
                frames_batch = []
                frame_indices = []

        
        # Process remaining frames
        if frames_batch:
            face_results = self._batch_detect_faces(frames_batch, frame_indices, face_threshold)
            scene_results = self._batch_detect_scenes(frames_batch, frame_indices, scene_threshold)
            nsfw_results = self._batch_detect(frames_batch, frame_indices, 'nsfw', nsfw_threshold)
            violence_results = self._batch_detect(frames_batch, frame_indices, 'violence', violence_threshold)
            
            results['faces'].extend(face_results)
            results['scenes'].extend(scene_results)
            results['nsfw'].extend([res for res in nsfw_results if res['is_detected']])
            results['violence'].extend([res for res in violence_results if res['is_detected']])
        
        # Calculate statistics
        processing_time = time.time() - start_time
        results['stats']['processing_time'] = processing_time
        results['stats']['processing_fps'] = results['stats']['processed_frames'] / processing_time
        results['stats']['total_faces'] = sum(len(frame['faces']) for frame in results['faces'])
        results['stats']['total_scenes'] = sum(len(frame['objects']) for frame in results['scenes'])
        results['stats']['total_nsfw'] = len(results['nsfw'])
        results['stats']['total_violence'] = len(results['violence'])
        results['stats']['total_smoking'] = len(results['smoking'])
        #音频分析

        print("\nStarting audio analysis...")
        audio_results, segmented_audio_texts = self.process_audio(video_path)
        print('音频分析结果:',segmented_audio_texts)
        results['video_text'] = segmented_audio_texts
        if audio_results:
            sensitive_words = self.expand_to_frame_records(audio_results, fps, sample_rate)
            results['sensitive_words'] = sensitive_words
            results['stats']['total_sensitive_words'] = len(sensitive_words)
            print(f"Detected {len(sensitive_words)} sensitive word frames (sample_rate={sample_rate})")      

        #OCR识别进程
        print("\nStarting OCR analysis...")
        try:
            ocr_results = self.process_video(video_path, batch_size=batch_size, frame_interval=sample_rate)
            
            if ocr_results:
                # 保存敏感词OCR结果
                results['ocr'] = ocr_results['sensitive_ocr']
                results['stats']['total_ocr_texts'] = sum(len(frame['text']) for frame in ocr_results['sensitive_ocr'])
                
                # 保存所有OCR的分段结果
                results['all_ocr_segments'] = ocr_results['all_ocr_segments']
                
                print(f"Detected {results['stats']['total_ocr_texts']} sensitive text instances")
                print(f"Total OCR segments: {len(ocr_results['all_ocr_segments'])}")
        except Exception as e:
            print(f"OCR processing failed: {str(e)}")
        # Release resources
        cap.release()
        pbar.close()
        
        # Save results
        self._save_results(results)
        self._print_summary(results)
        
        # 生成分段汇总结果
        segmented_results = self._create_segmented_results(results)
        
        # 保存分段结果到JSON文件
        segmented_json_path = os.path.join(self.OUTPUT_DIRS['logs'], 'segmented_results.json')
        with open(segmented_json_path, 'w', encoding='utf-8') as f:
            json.dump(segmented_results, f, ensure_ascii=False, indent=2)
        print(f"Segmented results saved to: {segmented_json_path}")
        
        # Store last results for API
        self.last_results = results
        self.last_segmented_results = segmented_results
        
        return results


    def _batch_detect_faces(
        self, 
        frames: List, 
        frame_indices: List[int], 
        confidence_threshold: float
    ) -> List[Dict]:
        """
        Batch face detection function
        
        Args:
            frames: List of frames
            frame_indices: Corresponding frame indices
            confidence_threshold: Confidence threshold
            
        Returns:
            List of detection results
        """
        # Perform batch prediction
        batch_results = self.models['face'].predict(
            source=frames,
            conf=confidence_threshold,
            save=False,
            verbose=False,
            imgsz=640,
            half=True,
            device=self.device,
            stream=False,
            augment=False
        )
        
        results = []
        
        for i, result in enumerate(batch_results):
            frame_result = {
                'frame': frame_indices[i],
                'timestamp': frame_indices[i] / self.models['face'].fps if hasattr(self.models['face'], 'fps') else 0,
                'faces': [],
                'image_path': ''
            }
            
            # Analyze detection results
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            
              
                # 与人脸库比对
                matched_person, similarity = self._compare_with_face_db(frames[i])
                
                face_info = {
                    'class_id': cls_id,
                    'class_name': self.models['face'].names[cls_id],
                    'confidence': conf,
                    'bbox': bbox,
                    'matched_person': matched_person,  # 匹配到的人名
                    'similarity': similarity  # 相似度
                }
                
                # 只有匹配到人脸库中的人脸才记录
                if matched_person:
                    frame_result['faces'].append(face_info)
            
            # Save annotated image (only if faces detected)
            if frame_result['faces']:
                filename = f"face_{frame_indices[i]:06d}.jpg"
                save_path = os.path.join(self.OUTPUT_DIRS['faces'], filename)
                # 直接保存原始帧（注意：frames中的图像是RGB格式，需要转为BGR）
                cv2.imwrite(save_path, cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
                frame_result['image_path'] = save_path
            
            results.append(frame_result)
        
        return results
    def _get_scene_chinese_name(self, english_name: str) -> str:
        """将英文场景名称转换为中文名称"""
        scene_name_mapping = {
            # A
            'airfield': '机场',
            'airplane_cabin': '飞机客舱',
            'airport_terminal': '航站楼',
            'alcove': '凹室',
            'alley': '小巷',
            'amphitheater': '圆形剧场',
            'amusement_arcade': '游戏厅',
            'amusement_park': '游乐园',
            'apartment_building/outdoor': '公寓楼/室外',
            'aquarium': '水族馆',
            'aqueduct': '渡槽',
            'arcade': '拱廊',
            'arch': '拱门',
            'archaelogical_excavation': '考古挖掘',
            'archive': '档案馆',
            'arena/hockey': '竞技场/冰球',
            'arena/performance': '竞技场/表演',
            'arena/rodeo': '竞技场/牛仔竞技',
            'army_base': '军事基地',
            'art_gallery': '艺术画廊',
            'art_school': '艺术学校',
            'art_studio': '艺术工作室',
            'artists_loft': '艺术家阁楼',
            'assembly_line': '装配线',
            'athletic_field/outdoor': '运动场/室外',
            'atrium/public': '中庭/公共',
            'attic': '阁楼',
            'auditorium': '礼堂',
            'auto_factory': '汽车工厂',
            'auto_showroom': '汽车展厅',
            
            # B
            'badlands': '荒地',
            'bakery/shop': '面包店',
            'balcony/exterior': '阳台/外部',
            'balcony/interior': '阳台/内部',
            'ball_pit': '球池',
            'ballroom': '舞厅',
            'bamboo_forest': '竹林',
            'bank_vault': '银行金库',
            'banquet_hall': '宴会厅',
            'bar': '酒吧',
            'barn': '谷仓',
            'barndoor': '仓门',
            'baseball_field': '棒球场',
            'basement': '地下室',
            'basketball_court/indoor': '篮球场/室内',
            'bathroom': '浴室',
            'bazaar/indoor': '集市/室内',
            'bazaar/outdoor': '集市/室外',
            'beach': '海滩',
            'beach_house': '海滩房屋',
            'beauty_salon': '美容院',
            'bedchamber': '卧室',
            'bedroom': '卧室',
            'beer_garden': '啤酒花园',
            'beer_hall': '啤酒厅',
            'berth': '泊位',
            'biology_laboratory': '生物实验室',
            'boardwalk': '木板路',
            'boat_deck': '船甲板',
            'boathouse': '船库',
            'bookstore': '书店',
            'booth/indoor': '摊位/室内',
            'botanical_garden': '植物园',
            'bow_window/indoor': '凸窗/室内',
            'bowling_alley': '保龄球馆',
            'boxing_ring': '拳击台',
            'bridge': '桥梁',
            'building_facade': '建筑立面',
            'bullring': '斗牛场',
            'burial_chamber': '墓室',
            'bus_interior': '公交车内部',
            'bus_station/indoor': '公交车站/室内',
            'butchers_shop': '肉铺',
            'butte': '孤峰',
            
            # C
            'cabin/outdoor': '小屋/室外',
            'cafeteria': '自助餐厅',
            'campsite': '露营地',
            'campus': '校园',
            'canal/natural': '运河/自然',
            'canal/urban': '运河/城市',
            'candy_store': '糖果店',
            'canyon': '峡谷',
            'car_interior': '汽车内部',
            'carrousel': '旋转木马',
            'castle': '城堡',
            'catacomb': '地下墓穴',
            'cemetery': '墓地',
            'chalet': '木屋',
            'chemistry_lab': '化学实验室',
            'childs_room': '儿童房',
            'church/indoor': '教堂/室内',
            'church/outdoor': '教堂/室外',
            'classroom': '教室',
            'clean_room': '洁净室',
            'cliff': '悬崖',
            'closet': '壁橱',
            'clothing_store': '服装店',
            'coast': '海岸',
            'cockpit': '驾驶舱',
            'coffee_shop': '咖啡店',
            'computer_room': '计算机房',
            'conference_center': '会议中心',
            'conference_room': '会议室',
            'construction_site': '建筑工地',
            'corn_field': '玉米田',
            'corral': '畜栏',
            'corridor': '走廊',
            'cottage': '村舍',
            'courthouse': '法院',
            'courtyard': '庭院',
            'creek': '小溪',
            'crevasse': '冰裂缝',
            'crosswalk': '人行横道',
            
            # D
            'dam': '水坝',
            'delicatessen': '熟食店',
            'department_store': '百货商店',
            'desert/sand': '沙漠/沙地',
            'desert/vegetation': '沙漠/植被',
            'desert_road': '沙漠公路',
            'diner/outdoor': '小餐馆/室外',
            'dining_hall': '餐厅',
            'dining_room': '餐厅',
            'discotheque': '迪斯科舞厅',
            'doorway/outdoor': '门口/室外',
            'dorm_room': '宿舍房间',
            'downtown': '市中心',
            'dressing_room': '更衣室',
            'driveway': '车道',
            'drugstore': '药店',
            
            # E
            'elevator/door': '电梯/门',
            'elevator_lobby': '电梯厅',
            'elevator_shaft': '电梯井',
            'embassy': '大使馆',
            'engine_room': '发动机房',
            'entrance_hall': '入口大厅',
            'escalator/indoor': '自动扶梯/室内',
            'excavation': '挖掘现场',
            
            # F
            'fabric_store': '布料店',
            'farm': '农场',
            'fastfood_restaurant': '快餐店',
            'field/cultivated': '田野/耕地',
            'field/wild': '田野/野生',
            'field_road': '田间道路',
            'fire_escape': '消防通道',
            'fire_station': '消防站',
            'fishpond': '鱼塘',
            'flea_market/indoor': '跳蚤市场/室内',
            'florist_shop/indoor': '花店/室内',
            'food_court': '美食广场',
            'football_field': '足球场',
            'forest/broadleaf': '森林/阔叶林',
            'forest_path': '森林小径',
            'forest_road': '森林道路',
            'formal_garden': '正规花园',
            'fountain': '喷泉',
            
            # G
            'galley': '厨房',
            'garage/indoor': '车库/室内',
            'garage/outdoor': '车库/室外',
            'gas_station': '加油站',
            'gazebo/exterior': '凉亭/外部',
            'general_store/indoor': '杂货店/室内',
            'general_store/outdoor': '杂货店/室外',
            'gift_shop': '礼品店',
            'glacier': '冰川',
            'golf_course': '高尔夫球场',
            'greenhouse/indoor': '温室/室内',
            'greenhouse/outdoor': '温室/室外',
            'grotto': '洞穴',
            'gymnasium/indoor': '体育馆/室内',
            
            # H
            'hangar/indoor': '机库/室内',
            'hangar/outdoor': '机库/室外',
            'harbor': '港口',
            'hardware_store': '五金店',
            'hayfield': '干草田',
            'heliport': '直升机停机坪',
            'highway': '高速公路',
            'home_office': '家庭办公室',
            'home_theater': '家庭影院',
            'hospital': '医院',
            'hospital_room': '医院病房',
            'hot_spring': '温泉',
            'hotel/outdoor': '酒店/室外',
            'hotel_room': '酒店房间',
            'house': '房屋',
            'hunting_lodge/outdoor': '狩猎小屋/室外',
            
            # I
            'ice_cream_parlor': '冰淇淋店',
            'ice_floe': '浮冰',
            'ice_shelf': '冰架',
            'ice_skating_rink/indoor': '溜冰场/室内',
            'ice_skating_rink/outdoor': '溜冰场/室外',
            'iceberg': '冰山',
            'igloo': '冰屋',
            'industrial_area': '工业区',
            'inn/outdoor': '旅馆/室外',
            'islet': '小岛',
            
            # J
            'jacuzzi/indoor': '按摩浴缸/室内',
            'jail_cell': '监狱牢房',
            'japanese_garden': '日式花园',
            'jewelry_shop': '珠宝店',
            'junkyard': '废品场',
            
            # K
            'kasbah': '卡斯巴',
            'kennel/outdoor': '狗舍/室外',
            'kindergarden_classroom': '幼儿园教室',
            'kitchen': '厨房',
            
            # L
            'lagoon': '泻湖',
            'lake/natural': '湖泊/自然',
            'landfill': '垃圾填埋场',
            'landing_deck': '着陆甲板',
            'laundromat': '自助洗衣店',
            'lawn': '草坪',
            'lecture_room': '演讲厅',
            'legislative_chamber': '立法会议厅',
            'library/indoor': '图书馆/室内',
            'library/outdoor': '图书馆/室外',
            'lighthouse': '灯塔',
            'living_room': '客厅',
            'loading_dock': '装卸码头',
            'lobby': '大堂',
            'lock_chamber': '船闸室',
            'locker_room': '更衣室',
            
            # M
            'mansion': '豪宅',
            'manufactured_home': '预制房屋',
            'market/indoor': '市场/室内',
            'market/outdoor': '市场/室外',
            'marsh': '沼泽',
            'martial_arts_gym': '武术馆',
            'mausoleum': '陵墓',
            'medina': '麦地那',
            'mezzanine': '中层楼',
            'moat/water': '护城河/水',
            'mosque/outdoor': '清真寺/室外',
            'motel': '汽车旅馆',
            'mountain': '山脉',
            'mountain_path': '山路',
            'mountain_snowy': '雪山',
            'movie_theater/indoor': '电影院/室内',
            'museum/indoor': '博物馆/室内',
            'museum/outdoor': '博物馆/室外',
            'music_studio': '音乐工作室',
            
            # N
            'natural_history_museum': '自然历史博物馆',
            'nursery': '托儿所',
            'nursing_home': '养老院',
            
            # O
            'oast_house': '烘干房',
            'ocean': '海洋',
            'office': '办公室',
            'office_building': '办公楼',
            'office_cubicles': '办公室隔间',
            'oilrig': '石油钻井平台',
            'operating_room': '手术室',
            'orchard': '果园',
            'orchestra_pit': '乐池',
            
            # P
            'pagoda': '宝塔',
            'palace': '宫殿',
            'pantry': '食品储藏室',
            'park': '公园',
            'parking_garage/indoor': '停车场/室内',
            'parking_garage/outdoor': '停车场/室外',
            'parking_lot': '停车场',
            'pasture': '牧场',
            'patio': '露台',
            'pavilion': '亭子',
            'pet_shop': '宠物店',
            'pharmacy': '药店',
            'phone_booth': '电话亭',
            'physics_laboratory': '物理实验室',
            'picnic_area': '野餐区',
            'pier': '码头',
            'pizzeria': '比萨店',
            'playground': '游乐场',
            'playroom': '游戏室',
            'plaza': '广场',
            'pond': '池塘',
            'porch': '门廊',
            'promenade': '步行道',
            'pub/indoor': '酒吧/室内',
            
            # R
            'racecourse': '赛马场',
            'raceway': '赛道',
            'raft': '木筏',
            'railroad_track': '铁路轨道',
            'rainforest': '雨林',
            'reception': '接待处',
            'recreation_room': '娱乐室',
            'repair_shop': '修理店',
            'residential_neighborhood': '住宅区',
            'restaurant': '餐厅',
            'restaurant_kitchen': '餐厅厨房',
            'restaurant_patio': '餐厅露台',
            'rice_paddy': '稻田',
            'river': '河流',
            'rock_arch': '石拱门',
            'roof_garden': '屋顶花园',
            'rope_bridge': '绳桥',
            'ruin': '废墟',
            'runway': '跑道',
            
            # S
            'sandbox': '沙箱',
            'sauna': '桑拿房',
            'schoolhouse': '校舍',
            'science_museum': '科学博物馆',
            'server_room': '服务器机房',
            'shed': '棚屋',
            'shoe_shop': '鞋店',
            'shopfront': '店铺门面',
            'shopping_mall/indoor': '购物中心/室内',
            'shower': '淋浴间',
            'ski_resort': '滑雪胜地',
            'ski_slope': '滑雪坡',
            'sky': '天空',
            'skyscraper': '摩天大楼',
            'slum': '贫民窟',
            'snowfield': '雪原',
            'soccer_field': '足球场',
            'stable': '马厩',
            'stadium/baseball': '体育场/棒球',
            'stadium/football': '体育场/足球',
            'stadium/soccer': '体育场/足球',
            'stage/indoor': '舞台/室内',
            'stage/outdoor': '舞台/室外',
            'staircase': '楼梯',
            'storage_room': '储藏室',
            'street': '街道',
            'subway_station/platform': '地铁站/站台',
            'supermarket': '超市',
            'sushi_bar': '寿司店',
            'swamp': '沼泽',
            'swimming_hole': '游泳洞',
            'swimming_pool/indoor': '游泳池/室内',
            'swimming_pool/outdoor': '游泳池/室外',
            'synagogue/outdoor': '犹太教堂/室外',
            
            # T
            'television_room': '电视房',
            'television_studio': '电视演播室',
            'temple/asia': '寺庙/亚洲',
            'throne_room': '王座厅',
            'ticket_booth': '售票亭',
            'topiary_garden': '修剪花园',
            'tower': '塔楼',
            'toyshop': '玩具店',
            'train_interior': '火车内部',
            'train_station/platform': '火车站/站台',
            'tree_farm': '树场',
            'tree_house': '树屋',
            'trench': '战壕',
            'tundra': '苔原',
            
            # U
            'underwater/ocean_deep': '水下/深海',
            'utility_room': '杂物间',
            
            # V
            'valley': '山谷',
            'vegetable_garden': '菜园',
            'veterinarians_office': '兽医诊所',
            'viaduct': '高架桥',
            'village': '村庄',
            'vineyard': '葡萄园',
            'volcano': '火山',
            'volleyball_court/outdoor': '排球场/室外',
            
            # W
            'waiting_room': '等候室',
            'water_park': '水上乐园',
            'water_tower': '水塔',
            'waterfall': '瀑布',
            'watering_hole': '水坑',
            'wave': '波浪',
            'wet_bar': '湿酒吧',
            'wheat_field': '麦田',
            'wind_farm': '风电场',
            'windmill': '风车',
            
            # Y
            'yard': '院子',
            'youth_hostel': '青年旅社',
            
            # Z
            'zen_garden': '禅意花园'
        }
            
        # 如果找不到对应的中文名称，返回英文名称
        return scene_name_mapping.get(english_name, english_name)
    
    def _batch_detect_scenes(
        self, 
        frames: List, 
        frame_indices: List[int], 
        confidence_threshold: float
    ) -> List[Dict]:
        """
        Batch scene detection using ResNet50-places365
        
        Args:
            frames: List of frames
            frame_indices: Corresponding frame indices
            confidence_threshold: Confidence threshold
            
        Returns:
            List of detection results
        """
        if self.scene_model is None:
            return [{'frame': idx, 'timestamp': idx/30, 'objects': [], 'image_path': ''} 
                    for idx in frame_indices]
        
        results = []
        
        for i, frame in enumerate(frames):
            frame_result = {
                'frame': frame_indices[i],
                'timestamp': frame_indices[i] / 30,  # 使用估计的FPS
                'objects': [],
                'image_path': ''
            }
            
            try:
                # 转换图像格式
                pil_image = Image.fromarray(frame)
                
                # 预处理
                input_tensor = self.scene_transform(pil_image).unsqueeze(0).to(self.device)
                
                # 推理
                with torch.no_grad():
                    logits = self.scene_model(input_tensor)
                    probs = F.softmax(logits, 1)
                    top_probs, top_indices = probs.topk(8)  # 取前8个预测
                
                # 处理结果
                for prob, idx in zip(top_probs[0], top_indices[0]):
                    if prob >= confidence_threshold:
                        class_name = self.scene_classes[idx.item()]
                        chinese_name = self._get_scene_chinese_name(class_name)
                        obj_info = {
                            'class_id': idx.item(),
                            'class_name': class_name,
                            'chinese_name': chinese_name,
                            'confidence': prob.item(),
                            'bbox': [0, 0, 0, 0]  # ResNet不提供边界框
                        }
                        frame_result['objects'].append(obj_info)
                
                # 保存带标注的图像（如果检测到场景）
                if frame_result['objects']:
                    # 直接使用原始帧（注意：frames中的图像是RGB格式，保存时需转BGR）
                    filename = f"scene_{frame_indices[i]:06d}.jpg"
                    save_path = os.path.join(self.OUTPUT_DIRS['scenes'], filename)
                    # 转换为BGR格式后保存（OpenCV默认BGR）
                    cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    frame_result['image_path'] = save_path
                    
            except Exception as e:
                print(f"Scene detection error on frame {frame_indices[i]}: {str(e)}")
            
            results.append(frame_result)
        
        return results
    def _batch_detect_smoking(
        self, 
        frames: List, 
        frame_indices: List[int], 
        confidence_threshold: float
    ) -> List[Dict]:
        """
        Batch smoking detection function
        
        Args:
            frames: List of frames
            frame_indices: Corresponding frame indices
            confidence_threshold: Confidence threshold
            
        Returns:
            List of detection results
        """
        # Perform batch prediction
        batch_results = self.models['smoking'].predict(
            source=frames,
            conf=confidence_threshold,
            save=False,
            verbose=False,
            imgsz=640,
            half=True,
            device=self.device,
            stream=False,
            augment=False
        )
        
        results = []
        
        for i, result in enumerate(batch_results):
            detections = {
                'is_detected': False,
                'frame': frame_indices[i],
                'timestamp': frame_indices[i] / self.models['smoking'].fps if hasattr(self.models['smoking'], 'fps') else 0,
                'classes': [],
                'chinese_classes': [],
                'max_confidence': 0,
                'image_path': ''
            }
            
            # Analyze detection results
            for box in result.boxes:
                cls_id = int(box.cls)
                cls_name = self.models['smoking'].names[cls_id]
                conf = float(box.conf)
                
                if conf >= confidence_threshold:
                    detections['is_detected'] = True
                    detections['classes'].append(cls_name)
                    # 根据提供的类别信息，只有 'smoke' 类别
                    chinese_name = '抽烟'  # 将 'smoke' 映射为中文 '抽烟'
                    detections['chinese_classes'].append(chinese_name)
                    detections['max_confidence'] = max(detections['max_confidence'], conf)
            
            # Save annotated image (only if smoking detected)
            if detections['is_detected']:
                filename = f"smoking_{frame_indices[i]:06d}.jpg"
                save_path = os.path.join(self.OUTPUT_DIRS['smoking'], filename)
                # 直接保存原始帧
                cv2.imwrite(save_path, cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
                detections['image_path'] = save_path
            
            results.append(detections)
        
        return results

    def _batch_detect(
        self, 
        frames: List, 
        frame_indices: List[int], 
        detection_type: str, 
        confidence_threshold: float
    ) -> List[Dict]:
        """
        Batch detection function (supports NSFW and violence detection)
        
        Args:
            frames: List of frames
            frame_indices: Corresponding frame indices
            detection_type: Detection type ('nsfw' or 'violence')
            confidence_threshold: Confidence threshold
            
        Returns:
            List of detection results
        """
        if detection_type not in ['nsfw', 'violence']:
            raise ValueError("Detection type must be 'nsfw' or 'violence'")
        
        # Perform batch prediction
        batch_results = self.models[detection_type].predict(
            source=frames,
            conf=confidence_threshold,
            save=False,
            verbose=False,
            imgsz=640,
            half=True,
            device=self.device,
            stream=False,
            augment=False  # Disable augmentation for speed
        )
        
        results = []
        class_mapping = self.NSFW_CLASSES if detection_type == 'nsfw' else self.VIOLENCE_CLASSES
        label_set = self.NSFW_LABELS if detection_type == 'nsfw' else self.VIOLENCE_LABELS
        
        for i, result in enumerate(batch_results):
            detections = {
                'is_detected': False,
                'frame': frame_indices[i],
                'timestamp': frame_indices[i] / self.models[detection_type].fps if hasattr(self.models[detection_type], 'fps') else 0,
                'classes': [],
                'chinese_classes': [],
                'max_confidence': 0,
                'image_path': ''
            }
            
            # Analyze detection results
            for box in result.boxes:
                cls_name = self.models[detection_type].names[int(box.cls)]
                conf = float(box.conf)
                
                if cls_name in label_set and conf >= confidence_threshold:
                    detections['is_detected'] = True
                    detections['classes'].append(cls_name)
                    detections['chinese_classes'].append(class_mapping.get(cls_name, cls_name))
                    detections['max_confidence'] = max(detections['max_confidence'], conf)
            
            # Save annotated image (only if content detected)
            if detections['is_detected']:
                filename = f"{detection_type}_{frame_indices[i]:06d}.jpg"
                save_path = os.path.join(self.OUTPUT_DIRS[detection_type], filename)
                # 直接保存原始帧
                cv2.imwrite(save_path, cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
                detections['image_path'] = save_path
            
            results.append(detections)
        
        return results

    def _save_results(self, results: Dict) -> None:
        """Save all results to CSV files"""
        # Save face detection records
        face_csv_path = os.path.join(self.OUTPUT_DIRS['logs'], 'face_detections.csv')
        with open(face_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Timestamp(s)', 'Face Count', 'Person', 'Similarity', 'Face Class', 'Confidence', 'Bounding Box', 'Image Path'])
            for face in results['faces']:
                if face['faces']:
                    for f in face['faces']:
                        writer.writerow([
                            face['frame'],
                            f"{face['timestamp']:.2f}",
                            len(face['faces']),
                            f.get('matched_person', '未知'),
                            f"{f.get('similarity', 0):.4f}",
                            f['class_name'],
                            f"{f['confidence']:.4f}",
                            f"{f['bbox']}",
                            face['image_path']
                        ])
        
        # 保存场景监测
        scene_csv_path = os.path.join(self.OUTPUT_DIRS['logs'], 'scene_detections.csv')
        with open(scene_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Timestamp(s)', 'Scene Count', 'English Name', 'Chinese Name', 'Confidence', 'Image Path'])
            for scene in results['scenes']:
                if scene['objects']:
                    for s in scene['objects']:
                        writer.writerow([
                            scene['frame'],
                            f"{scene['timestamp']:.2f}",
                            len(scene['objects']),
                            s['class_name'],
                            s['chinese_name'],
                            f"{s['confidence']:.4f}",
                            scene['image_path']
                        ])
        
        # 保存色情监测
        nsfw_csv_path = os.path.join(self.OUTPUT_DIRS['logs'], 'nsfw_detections.csv')
        with open(nsfw_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Timestamp(s)', 'English Classes', 'Chinese Classes', 'Max Confidence', 'Image Path'])
            for nsfw in results['nsfw']:
                writer.writerow([
                    nsfw['frame'],
                    f"{nsfw['timestamp']:.2f}",
                    ','.join(nsfw['classes']),
                    ','.join(nsfw['chinese_classes']),
                    f"{nsfw['max_confidence']:.4f}",
                    nsfw['image_path']
                ])
        
        # 保存暴力监测
        violence_csv_path = os.path.join(self.OUTPUT_DIRS['logs'], 'violence_detections.csv')
        with open(violence_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Timestamp(s)', 'English Classes', 'Chinese Classes', 'Max Confidence', 'Image Path'])
            for violence in results['violence']:
                writer.writerow([
                    violence['frame'],
                    f"{violence['timestamp']:.2f}",
                    ','.join(violence['classes']),
                    ','.join(violence['chinese_classes']),
                    f"{violence['max_confidence']:.4f}",
                    violence['image_path']
                ])
        # 保存抽烟检测记录
        smoking_csv_path = os.path.join(self.OUTPUT_DIRS['logs'], 'smoking_detections.csv')
        with open(smoking_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Timestamp(s)', 'English Classes', 'Chinese Classes', 'Max Confidence', 'Image Path'])
            for smoking in results['smoking']:
                writer.writerow([
                    smoking['frame'],
                    f"{smoking['timestamp']:.2f}",
                    ','.join(smoking['classes']),
                    ','.join(smoking['chinese_classes']),
                    f"{smoking['max_confidence']:.4f}",
                    smoking['image_path']
                ])
        # 保存黑场帧记录
        black_csv_path = os.path.join(self.OUTPUT_DIRS['logs'], 'black_frames.csv')
        with open(black_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Frame', 
                'Timestamp(s)', 
                'Type', 
                'Chinese Type', 
                'Confidence', 
                'Segment Start Frame', 
                'Segment Duration Frames'
            ])
            
            for frame in results['abnormalities']['black_frames']:
                writer.writerow([
                    frame['frame'],
                    f"{frame['timestamp']:.2f}",
                    frame['type'],
                    frame['chinese_type'],
                    f"{frame['confidence']:.4f}",
                    frame['segment_start_frame'],
                    frame['segment_duration_frames']
                ])
        #保存静止帧监测
        frozen_csv_path = os.path.join(self.OUTPUT_DIRS['logs'], 'frozen_frames.csv')
        with open(frozen_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Frame', 
                'Timestamp(s)', 
                'Type', 
                'Chinese Type', 
                'Confidence', 
                'Segment Start Frame', 
                'Segment Duration Frames'
            ])
            
            for frame in results['abnormalities']['frozen_frames']:
                writer.writerow([
                    frame['frame'],
                    f"{frame['timestamp']:.2f}",
                    frame['type'],
                    frame['chinese_type'],
                    f"{frame['confidence']:.4f}",
                    frame['segment_start_frame'],
                    frame['segment_duration_frames']
                ])
        # 保存敏感词检测结果(考虑采样率)
        sensitive_csv_path = os.path.join(self.OUTPUT_DIRS['logs'], 'sensitive_words.csv')
        with open(sensitive_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Frame', 
                'Timestamp(s)', 
                'Start Time', 
                'End Time', 
                'Frame Range',
                'Sample Rate',
                'Text', 
                'Sensitive Words'
            ])
            for item in results['sensitive_words']:
                writer.writerow([
                    item['frame'],
                    f"{item['timestamp']:.3f}",
                    item['start_time'],
                    item['end_time'],
                    item['frame_range'],
                    item.get('sample_rate', 1),
                    item['text'],
                    ','.join(item['words'])
                ])

        # 保存OCR识别结果
        ocr_csv_path = os.path.join(self.OUTPUT_DIRS['logs'], 'ocr_detections.csv')
        with open(ocr_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Timestamp(s)', 'Text Count', 'Text Content'])
            for ocr_frame in results['ocr']:
                if ocr_frame['text']:
                    writer.writerow([
                        ocr_frame['frame_num'],
                        f"{ocr_frame['frame_num'] / results['video_info']['fps']:.2f}",
                        len(ocr_frame['text']),
                        '|'.join(ocr_frame['text'])  # 用竖线分隔多个文本
                    ])
        
        # Save statistics
        stats_csv_path = os.path.join(self.OUTPUT_DIRS['logs'], 'analysis_stats.csv')
        with open(stats_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Statistic', 'Value'])
            writer.writerow(['Video Path', results['video_info']['path']])
            writer.writerow(['Video Duration(s)', f"{results['video_info']['duration']:.2f}"])
            writer.writerow(['Total Frames', results['video_info']['total_frames']])
            writer.writerow(['Processed Frames', results['stats']['processed_frames']])
            writer.writerow(['Face Detections', results['stats']['total_faces']])
            writer.writerow(['Scene Objects', results['stats']['total_scenes']])
            writer.writerow(['NSFW Detections', results['stats']['total_nsfw']])
            writer.writerow(['Violence Detections', results['stats']['total_violence']])
            writer.writerow(['Processing Time(s)', f"{results['stats']['processing_time']:.2f}"])
            writer.writerow(['Processing Speed(FPS)', f"{results['stats']['processing_fps']:.2f}"])
            writer.writerow(['Sample Rate', results['analysis_params']['sample_rate']])
            writer.writerow(['Batch Size', results['analysis_params']['batch_size']])
            writer.writerow(['OCR Text Detections', results['stats']['total_ocr_texts']])

    def _print_summary(self, results: Dict) -> None:
        """Print analysis summary"""
        stats = results['stats']
        print("\n===== Analysis Summary =====")
        print(f"Video Path: {results['video_info']['path']}")
        print(f"Video Duration: {results['video_info']['duration']:.2f}s")
        print(f"Total Frames: {results['video_info']['total_frames']}")
        print(f"Processed Frames: {stats['processed_frames']} (Sample rate: {results['analysis_params']['sample_rate']})")
        print(f"Face Detections: {stats['total_faces']}")
        print(f"Scene Objects: {stats['total_scenes']}")
        print(f"NSFW Detections: {stats['total_nsfw']}")
        print(f"Violence Detections: {stats['total_violence']}")
        print(f"\nProcessing Time: {stats['processing_time']:.2f}s")
        print(f"Processing Speed: {stats['processing_fps']:.2f} FPS")
        print(f"Batch Size: {results['analysis_params']['batch_size']}")
        print("\nResults saved to:")
        print(f"- Face Detections: {self.OUTPUT_DIRS['faces']}")
        print(f"- Scene Detections: {self.OUTPUT_DIRS['scenes']}")
        print(f"- NSFW Detections: {self.OUTPUT_DIRS['nsfw']}")
        print(f"- Violence Detections: {self.OUTPUT_DIRS['violence']}")
        print(f"- Log Files: {self.OUTPUT_DIRS['logs']}")
        print("="*30)
        print(f"Black Scenes: {results['stats']['black_scenes']}")
        print(f"Frozen Scenes: {results['stats']['frozen_scenes']}")
        print(f"OCR Text Detections: {results['stats']['total_ocr_texts']}")
    @classmethod
    def create_api_app(cls):
        """Create FastAPI application instance"""
        app = FastAPI(
            title="Video Content Analysis API",
            description="Provides video content analysis including face detection, scene recognition, NSFW and violence detection",
            version="1.0.0",
            debug=True
        )

        analyzer = cls()
        # 添加全局异常处理器（关键）
        @app.exception_handler(Exception)
        async def catch_all_exceptions(request, exc):
            import traceback
            # 获取完整的错误堆栈信息
            error_stack = traceback.format_exc()
            # 返回包含详细错误的响应
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error",
                    "message": "服务器内部错误",
                    "detail": str(exc),  # 错误描述
                    "stack_trace": error_stack  # 错误堆栈（方便调试）
                }
            )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # 允许所有来源（仅限开发环境）
            allow_credentials=True,
            allow_methods=["*"],  # 允许所有HTTP方法
            allow_headers=["*"],  # 允许所有HTTP头
        )
        @app.post("/analyze", summary="Analyze uploaded video file")
        async def analyze_video_api(
            file_path: Optional[str] = Form(None, description="Path to video on server (if exists)"),
            # file: UploadFile = File(..., description="Video file to analyze"),
            sample_rate: int = Form(1, description="Frame sampling rate (1=every frame)"),
            batch_size: int = Form(4, description="Batch processing size"),
            nsfw_threshold: float = Form(0.5, description="NSFW confidence threshold"),
            violence_threshold: float = Form(0.5, description="Violence confidence threshold"),
            face_threshold: float = Form(0.5, description="Face detection threshold"),
            scene_threshold: float = Form(0.3, description="Scene detection threshold"),
            frozen_threshold: int = Form(125),
            black_duration_threshold: int = Form(50),
            sensitive_words_path: List[str] = Form(None, description="List of paths to sensitive words dictionaries"),
            sensitive_faces_path: List[str] = Form(None, description="List of paths to sensitive faces databases"),
            whisper_model: str = Form("tiny", description="Whisper model size (tiny, base, small, medium, large)"),
            segment_duration: int = Form(5, description="Segment duration in seconds for result aggregation")
        ):
            """
            Analyze uploaded video file and return detection results
            
            - **file**: Video file to analyze (MP4, AVI etc.)
            - **sample_rate**: Process every N-th frame (default 1)
            - **batch_size**: Number of frames processed together (default 4)
            - **nsfw_threshold**: NSFW confidence threshold (0-1)
            - **violence_threshold**: Violence confidence threshold (0-1)
            - **face_threshold**: Face detection threshold (0-1)
            - **scene_threshold**: Scene detection threshold (0-1)
            - **segment_duration**: Duration of each segment in seconds for result aggregation (default 5)
            
            """
            # 校验：必须提供其中一种方式
            # if not file_path and not file:
            #     raise HTTPException(
            #         status_code=400,
            #         detail="Either 'file_path' (server local path) or 'file' (upload) must be provided"
            #     )
            
            # 处理服务器本地文件路径
            if file_path:
                print("使用本地文件路径进行分析:", file_path)
                if not os.path.exists(file_path):
                    raise HTTPException(
                        status_code=404,
                        detail=f"Server file not found: {file_path}"
                    )
                # 直接使用服务器上的文件路径
                video_path = file_path
            # else:
            #     # 处理上传的文件（原有逻辑）
            #     with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            #         content = await file.read()
            #         tmp.write(content)
            #         video_path = tmp.name
            def convert_paths(path_list):
                if not path_list:
                    return []
                # 如果收到的是 ['path1,path2'] 格式
                if len(path_list) == 1 and ',' in path_list[0]:
                    return [path.strip() for path in path_list[0].split(',') if path.strip()]
                # 如果收到的是 ['path1', 'path2'] 格式
                return [path.strip() for path in path_list if path.strip()]
            
            sensitive_words_list = convert_paths(sensitive_words_path)
            sensitive_faces_list = convert_paths(sensitive_faces_path)
            try:
                # Save uploaded temporary file
                # with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                #     content = await file.read()
                #     tmp.write(content)
                #     tmp_path = tmp.name
            
                # Call analysis method with all parameters
                results = analyzer.analyze_video(
                    video_path=video_path,
                    sample_rate=sample_rate,
                    batch_size=batch_size,
                    nsfw_threshold=nsfw_threshold,
                    violence_threshold=violence_threshold,
                    face_threshold=face_threshold,
                    scene_threshold=scene_threshold,
                    frozen_threshold=frozen_threshold,
                    black_duration_threshold=black_duration_threshold,
                    sensitive_words_path=sensitive_words_list,  
                    sensitive_faces_path=sensitive_faces_list,
                    whisper_model=whisper_model
                )
       
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
                # 生成分段汇总结果
                segmented_results = analyzer._create_segmented_results(results, segment_duration)
                
                # Prepare detailed detection results
                detection_results = {
                    "nsfw": [
                        {
                            "timestamp": det["timestamp"],
                            "frame": det["frame"],
                            "classes": det["chinese_classes"],
                            "confidence": det["max_confidence"],
                            "image_path": det["image_path"]
                        } for det in results["nsfw"]
                    ],
                    "violence": [
                        {
                            "timestamp": det["timestamp"],
                            "frame": det["frame"],
                            "classes": det["chinese_classes"],
                            "confidence": det["max_confidence"],
                            "image_path": det["image_path"]
                        } for det in results["violence"]
                    ],
                    # 在detection_results中添加
                    "smoking": [
                        {
                            "timestamp": det["timestamp"],
                            "frame": det["frame"],
                            "classes": det["chinese_classes"],
                            "confidence": det["max_confidence"],
                            "image_path": det["image_path"]
                        } for det in results["smoking"]
                    ],
                    "faces": [
                        {
                            "timestamp": frame["timestamp"],
                            "frame": frame["frame"],
                            "count": len(frame["faces"]),
                            "matched_persons": [f.get('matched_person') for f in frame["faces"]],
                            "similarities": [float(f.get("similarity", 0)) for f in frame["faces"]],
                            "image_path": frame["image_path"]
                        } for frame in results["faces"] if frame["faces"]
                    ],
                    # 新增黑场检测结果
                    "black_scenes": [
                        {
                            "timestamp": frame["timestamp"],
                            "frame": frame["frame"],
                            "type": frame["chinese_type"],
                            "confidence": frame["confidence"],
                            "segment_start_frame": frame["segment_start_frame"],
                            "segment_duration_frames": frame["segment_duration_frames"]
                        } for frame in results["abnormalities"]["black_frames"]
                    ],
                    #静止帧监测结果
                    "frozen_scenes": [
                        {
                            "timestamp": frame["timestamp"],
                            "frame": frame["frame"],
                            "type": frame["chinese_type"],
                            "confidence": frame["confidence"],
                            "segment_start_frame": frame["segment_start_frame"],
                            "segment_duration_frames": frame["segment_duration_frames"]
                        } for frame in results["abnormalities"]["frozen_frames"]
                    ],
                    #敏感词检测结果
                    "sensitive_words": [
                        {
                            "frame": word["frame"],
                            "timestamp": word["timestamp"],
                            "text": word["text"],
                            "words": word["words"],
                            "chinese_words": word["chinese_words"],
                            "frame_range": word["frame_range"],
                            "sample_rate": word["sample_rate"]  # 添加采样率信息
                        } for word in results["sensitive_words"]
                    ],
                    "scenes": [
                        {
                            "timestamp": scene["timestamp"],
                            "frame": scene["frame"],
                            "object_count": len(scene["objects"]),  # 使用 'objects'
                            "objects": [
                                {
                                    "class_name": obj["class_name"],
                                    "confidence": float(obj["confidence"]),
                                    "bbox": obj["bbox"]
                                } for obj in scene["objects"]  # 使用 'objects'
                            ],
                            "image_path": scene["image_path"]
                        } for scene in results["scenes"] if scene["objects"]  # 使用 'objects'
                    ],
                    #OCR检测结果
                    "ocr_texts": [
                        {
                            "frame": frame["frame_num"],
                            "timestamp": frame["frame_num"] / results["video_info"]["fps"],
                            "text_count": len(frame["text"]),
                            "texts": frame["text"]
                        } for frame in results["ocr"] if frame["text"]
                    ]
                    
                }

                return JSONResponse({
                    "status": "success",
                    "data": {
                        "video_info": results["video_info"],
                        "stats": results["stats"],
                        "detections": detection_results,
                        "segmented_results": segmented_results  # 新增分段汇总结果
                    },
                    "detail": "Analysis completed"
                })
                
            except Exception as e:
                print(f"❌ 分析过程中出错: {str(e)}")
                import traceback
                print(f"📋 详细堆栈:\n{traceback.format_exc()}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Video analysis failed: {str(e)}"
                )
        
        @app.get("/results/summary", summary="Get summary of last analysis results")
        async def get_last_summary():
            """Get statistical summary of the last analysis results"""
            if not hasattr(analyzer, "last_results"):
                raise HTTPException(
                    status_code=404,
                    detail="No analysis performed yet, no results available"
                )
            
            # Prepare detailed detection results
            last_results = analyzer.last_results
            detection_results = {
                "nsfw": [
                    {
                        "timestamp": det["timestamp"],
                        "frame": det["frame"],
                        "classes": det["chinese_classes"],
                        "confidence": det["max_confidence"],
                        "image_path": det["image_path"]
                    } for det in last_results["nsfw"]
                ],
                "violence": [
                    {
                        "timestamp": det["timestamp"],
                        "frame": det["frame"],
                        "classes": det["chinese_classes"],
                        "confidence": det["max_confidence"],
                        "image_path": det["image_path"]
                    } for det in last_results["violence"]
                ],
                "faces": [
                    {
                        "timestamp": frame["timestamp"],
                        "frame": frame["frame"],
                        "count": len(frame["faces"]),
                        "matched_persons": [f.get('matched_person') for f in frame["faces"]],
                        "similarities": [f.get('similarity', 0) for f in frame["faces"]],
                        "image_path": frame["image_path"]
                    } for frame in last_results["faces"] if frame["faces"]
                ],
                "black_scenes": [
                    {
                        "timestamp": frame["timestamp"],
                        "frame": frame["frame"],
                        "type": frame["chinese_type"],
                        "confidence": frame["confidence"],
                        "segment_start_frame": frame["segment_start_frame"],
                        "segment_duration_frames": frame["segment_duration_frames"]
                    } for frame in last_results["abnormalities"]["black_frames"]
                ],
                "frozen_scenes": [
                    {
                        "timestamp": frame["timestamp"],
                        "frame": frame["frame"],
                        "type": frame["chinese_type"],
                        "confidence": frame["confidence"],
                        "segment_start_frame": frame["segment_start_frame"],
                        "segment_duration_frames": frame["segment_duration_frames"]
                    } for frame in last_results["abnormalities"]["frozen_frames"]
                ],
                "sensitive_words": [
                    {
                        "frame": word["frame"],
                        "timestamp": word["timestamp"],
                        "text": word["text"],
                        "words": word["words"],
                        "chinese_words": word["chinese_words"],
                        "frame_range": word["frame_range"],
                        "sample_rate": word["sample_rate"]  # 添加采样率信息
                    } for word in last_results["sensitive_words"]
                ],
                "scenes": [
                    {
                        "timestamp": scene["timestamp"],
                        "frame": scene["frame"],
                        "object_count": len(scene["objects"]),
                        "objects": [
                            {
                                "class_name": obj["class_name"],

                                "confidence": float(obj["confidence"]),
                                "bbox": obj["bbox"]
                            } for obj in scene["objects"]
                        ],
                        "image_path": scene["image_path"]
                    } for scene in last_results["scenes"] if scene["objects"]
                ],
                "ocr_texts": [
                    {
                        "frame": frame["frame_num"],
                        "timestamp": frame["frame_num"] / last_results["video_info"]["fps"],
                        "text_count": len(frame["text"]),
                        "texts": frame["text"]
                    } for frame in last_results["ocr"] if frame["text"]
                ]
            }
            
            # 包含分段汇总结果
            segmented_results = getattr(analyzer, "last_segmented_results", {})
            
            return JSONResponse({
                "status": "success",
                "data": {
                    "video_info": last_results["video_info"],
                    "stats": last_results["stats"],
                    "detections": detection_results,
                    "segmented_results": segmented_results
                }
            })
        
        @app.get("/results/segmented", summary="Get segmented analysis results")
        async def get_segmented_results():
            """Get analysis results segmented by time intervals"""
            if not hasattr(analyzer, "last_segmented_results"):
                raise HTTPException(
                    status_code=404,
                    detail="No segmented results available"
                )
            
            return JSONResponse({
                "status": "success",
                "data": analyzer.last_segmented_results
            })
        
        return app

def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Start API server"""
    app = EnhancedVideoContentAnalyzer.create_api_app()
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Content Analysis Tool")
    parser.add_argument(
        "--mode", 
        choices=["cli", "api"],
        default="cli",
        help="Operation mode: cli(command line) or api(API service)"
    )
    parser.add_argument(
        "--video", 
        type=str,
        help="Path to video file to analyze (for CLI mode)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="API service host address"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API service port"
    )
    
    args = parser.parse_args()
    
    if args.mode == "cli":
        if not args.video:
            print("Error: CLI mode requires --video parameter")
            exit(1)
            
        analyzer = EnhancedVideoContentAnalyzer()
        analyzer.analyze_video(
            video_path=args.video,
            sample_rate=2,
            batch_size=16,
            nsfw_threshold=0.6,
            violence_threshold=0.7,
            face_threshold=0.5,
            scene_threshold=0.3
        )
    else:
        print(f"Starting API service on {args.host}:{args.port}")
        run_api_server(host=args.host, port=args.port)