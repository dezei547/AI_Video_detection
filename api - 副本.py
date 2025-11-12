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
            'scene': 'models/yolo11s.pt',
            'nsfw': 'models/NSFW_YOLOv8.pt',
            'violence': 'models/yolov8-violence.pt',
            'smoking': 'models/yolov8_smoking.pt'
        }
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
                "sensitive_words": [],
                "adult_content": False,
                "smoking": False,
                "violence": False,
                "black_scenes": False,
                "frozen_scenes": False,
                "matched_faces": [],
                "ocr_texts": []
            }
        
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
        
        # 处理OCR文本
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
                        if text not in segments[segment_key]['ocr_texts']:
                            segments[segment_key]['ocr_texts'].append(text)
        
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
                    "ocr_texts": []
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
        
        # 清理临时文件
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        
        return sensitive_results

    
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

    def process_batch(self, frames: List[np.ndarray], frame_numbers: List[int]) -> List[Optional[List[Dict[str, Any]]]]:
        """批量处理帧，只返回包含敏感词的OCR结果"""
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
                    # 检查是否包含敏感词
                    has_sensitive = any(word in text for word in self.sensitive_words)
                    if not has_sensitive:
                        continue
                        
                    scaled_box = [[int(x*2) for x in pt] for pt in box]
                    frame_results.append({
                        'text': text,
                        'score': float(score),
                        'bbox': scaled_box,
                        'frame_num': frame_num,
                        'timestamp': frame_num / 30
                    })
                
                # 只有包含敏感词的帧才返回结果
                results.append(frame_results if frame_results else None)
            except Exception as e:
                print(f"处理帧 {frame_num} 时出错: {str(e)}")
                results.append(None)
        return results

    def process_video(self, video_path: str, batch_size: int = 8, frame_interval: int = 10) -> List[Dict[str, Any]]:
        """处理视频文件（支持批量推理），只返回包含敏感词的OCR结果"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频文件: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"开始处理视频: {video_path} (总帧数: {total_frames}, FPS: {fps:.1f})")

        results_dict = {}
        frame_batch = []
        frame_num_batch = []
        frame_counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                if frame_batch:
                    batch_results = self.process_batch(frame_batch, frame_num_batch)
                    self._merge_results(batch_results, frame_num_batch, results_dict)
                break

            frame_counter += 1
            
            if frame_counter % frame_interval != 0:
                continue
                
            frame_batch.append(frame.copy())
            frame_num_batch.append(frame_counter)
            
            if len(frame_batch) == batch_size:
                batch_results = self.process_batch(frame_batch, frame_num_batch)
                self._merge_results(batch_results, frame_num_batch, results_dict)
                
                frame_batch = []
                frame_num_batch = []
                
                detected_texts = sum(len(texts) for texts in results_dict.values())
                print(f"已处理 {frame_counter}/{total_frames} 帧, 发现敏感文本 {detected_texts} 处")

        cap.release()
        
        # 转换为最终输出格式
        merged_results = [
            {'frame_num': frame_num, 'text': texts}
            for frame_num, texts in sorted(results_dict.items())
        ]
        
        return merged_results

    def _merge_results(self, batch_results: List[List[Dict]], frame_numbers: List[int], results_dict: Dict[int, List[str]]) -> None:
        """将批量结果合并到结果字典中，只保留包含敏感词的结果"""
        for frame_result, frame_num in zip(batch_results, frame_numbers):
            if frame_result is None:
                continue
                
            if frame_num not in results_dict:
                results_dict[frame_num] = []
                
            # 提取当前帧所有文本（已经过敏感词过滤）
            texts = [res['text'] for res in frame_result]
            results_dict[frame_num].extend(texts)
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
            models['scene'] = YOLO(self.MODEL_PATHS['scene']).to(self.device)
            models['nsfw'] = YOLO(self.MODEL_PATHS['nsfw']).to(self.device)
            models['violence'] = YOLO(self.MODEL_PATHS['violence']).to(self.device)
            models['smoking'] = YOLO(self.MODEL_PATHS['smoking']).to(self.device)  # 新增抽烟模型
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
            'ocr': [],  # 新增OCR识别结果
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
        audio_results = self.process_audio(video_path)
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
                results['ocr'] = ocr_results
                results['stats']['total_ocr_texts'] = sum(len(frame['text']) for frame in ocr_results)
                print(f"Detected {results['stats']['total_ocr_texts']} sensitive text instances")
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
                annotated_frame = result.plot()
                filename = f"face_{frame_indices[i]:06d}.jpg"
                save_path = os.path.join(self.OUTPUT_DIRS['faces'], filename)
                cv2.imwrite(save_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                frame_result['image_path'] = save_path
            
            results.append(frame_result)
        
        return results

    def _batch_detect_scenes(
        self, 
        frames: List, 
        frame_indices: List[int], 
        confidence_threshold: float
    ) -> List[Dict]:
        """
        Batch scene detection function
        
        Args:
            frames: List of frames
            frame_indices: Corresponding frame indices
            confidence_threshold: Confidence threshold
            
        Returns:
            List of detection results
        """
        # Perform batch prediction
        batch_results = self.models['scene'].predict(
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
                'timestamp': frame_indices[i] / self.models['scene'].fps if hasattr(self.models['scene'], 'fps') else 0,
                'objects': [],
                'image_path': ''
            }
            
            # Analyze detection results
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                
                obj_info = {
                    'class_id': cls_id,
                    'class_name': self.models['scene'].names[cls_id],
                    'confidence': conf,
                    'bbox': bbox
                }
                frame_result['objects'].append(obj_info)
            
            # Save annotated image (only if objects detected)
            if frame_result['objects']:
                annotated_frame = result.plot()
                filename = f"scene_{frame_indices[i]:06d}.jpg"
                save_path = os.path.join(self.OUTPUT_DIRS['scenes'], filename)
                cv2.imwrite(save_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                frame_result['image_path'] = save_path
            
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
                annotated_frame = result.plot()
                filename = f"smoking_{frame_indices[i]:06d}.jpg"
                save_path = os.path.join(self.OUTPUT_DIRS['smoking'], filename)
                cv2.imwrite(save_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
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
                annotated_frame = result.plot()
                filename = f"{detection_type}_{frame_indices[i]:06d}.jpg"
                save_path = os.path.join(self.OUTPUT_DIRS[detection_type], filename)
                cv2.imwrite(save_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
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
            writer.writerow(['Frame', 'Timestamp(s)', 'Object Count', 'Object Class', 'Confidence', 'Bounding Box', 'Image Path'])
            for scene in results['scenes']:
                if scene['objects']:
                    for obj in scene['objects']:
                        writer.writerow([
                            scene['frame'],
                            f"{scene['timestamp']:.2f}",
                            len(scene['objects']),
                            obj['class_name'],
                            f"{obj['confidence']:.4f}",
                            f"{obj['bbox']}",
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
            version="1.0.0"
        )
        analyzer = cls()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # 允许所有来源（仅限开发环境）
            allow_credentials=True,
            allow_methods=["*"],  # 允许所有HTTP方法
            allow_headers=["*"],  # 允许所有HTTP头
        )
        @app.post("/analyze", summary="Analyze uploaded video file")
        async def analyze_video_api(
            file: UploadFile = File(..., description="Video file to analyze"),
            sample_rate: int = Form(1, description="Frame sampling rate (1=every frame)"),
            batch_size: int = Form(4, description="Batch processing size"),
            nsfw_threshold: float = Form(0.5, description="NSFW confidence threshold"),
            violence_threshold: float = Form(0.5, description="Violence confidence threshold"),
            face_threshold: float = Form(0.5, description="Face detection threshold"),
            scene_threshold: float = Form(0.5, description="Scene detection threshold"),
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
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                    content = await file.read()
                    tmp.write(content)
                    tmp_path = tmp.name
            
                # Call analysis method with all parameters
                results = analyzer.analyze_video(
                    video_path=tmp_path,
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
            scene_threshold=0.5
        )
    else:
        print(f"Starting API service on {args.host}:{args.port}")
        run_api_server(host=args.host, port=args.port)