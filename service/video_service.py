import os
import logging
import pyttsx3
import ffmpeg
import random
import re
from typing import List
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class VideoService:
    def __init__(self):
        # 初始化TTS引擎
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # 语速
        self.tts_engine.setProperty('volume', 0.8)  # 音量

        # 设置中文语音（如果可用）
        voices = self.tts_engine.getProperty('voices')
        for voice in voices:
            if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break

    def clean_scene_description(self, text: str) -> str:
        """
        清理场景描述，移除"场景一"、"场景二"等场景提示词
        """
        # 移除场景提示词的正则表达式
        pattern = r'场景[一二三四五六七八九十\d]+[：:，,。.\s]*'
        cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # 清理其他可能的场景标识
        patterns = [
            r'第[一二三四五六七八九十\d]+个场景[：:，,。.\s]*',
            # r'第[一二三四五六七八九十\d]+幕[：:，,。.\s]*',
            # r'镜头[一二三四五六七八九十\d]+[：:，,。.\s]*',
            # r'画面[一二三四五六七八九十\d]+[：:，,。.\s]*'
        ]

        for pattern in patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)

        return cleaned_text.strip()

    def split_text_for_subtitles(self, text: str) -> List[str]:
        """
        将文本分割成适合字幕的小段，每段不超过20个字
        """
        text = text.strip()
        if not text:
            return []

        # 按标点符号分割
        sentences = re.split(r'[。！？；.!?;]', text)
        subtitles = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 如果句子长度超过20字，进一步分割
            while len(sentence) > 20:
                # 寻找合适的分割点（逗号、顿号等）
                split_point = -1
                for i in range(19, 0, -1):
                    if sentence[i] in '，、,':
                        split_point = i + 1
                        break

                if split_point == -1:
                    split_point = 20

                subtitles.append(sentence[:split_point])
                sentence = sentence[split_point:].strip()

            if sentence:
                subtitles.append(sentence)

        return subtitles

    def text_to_speech(self, text: str, output_path: str) -> float:
        """
        将文本转换为语音文件
        返回音频时长（秒）
        """
        try:
            # 清理场景描述
            clean_text = self.clean_scene_description(text)

            # 进一步清理文本，移除可能影响TTS的字符
            clean_text = clean_text.replace('\n', ' ').replace('\r', ' ').strip()

            # 生成语音文件
            self.tts_engine.save_to_file(clean_text, output_path)
            self.tts_engine.runAndWait()

            # 使用ffmpeg获取音频时长
            probe = ffmpeg.probe(output_path)
            duration = float(probe['streams'][0]['duration'])

            logger.info(f"语音生成成功: {output_path}, 时长: {duration:.2f}秒")
            return duration

        except Exception as e:
            logger.error(f"文本转语音失败: {str(e)}")
            # 如果TTS失败，返回默认时长
            return 3.0

    def get_image_dimensions(self, image_path: str) -> tuple:
        """获取图片尺寸"""
        try:
            with Image.open(image_path) as img:
                return img.size  # 返回 (width, height)
        except Exception as e:
            logger.error(f"获取图片尺寸失败: {str(e)}")
            return (1024, 512)  # 默认尺寸

    def create_image_with_effect(self, image_path: str, output_path: str, duration: float,
                                 target_width: int, target_height: int, effect_type: int = None):
        """
        为图片添加动态效果
        effect_type: 1=缓慢放大, 2=左到右, 3=上到下, 4=下到上, 5=右到左
        """
        if effect_type is None:
            effect_type = random.randint(1, 5)

        try:
            framerate = 25  # 25fps
            total_frames = int(duration * framerate)

            if effect_type == 1:
                # 缓慢放大效果
                scale_start = 1.0
                scale_end = 1.2

                (
                    ffmpeg
                    .input(image_path, loop=1, t=duration, framerate=framerate)
                    .filter('scale', target_width, target_height)
                    .filter('zoompan',
                            z=f'min(zoom+0.0015,{scale_end})',
                            x='iw/2-(iw/zoom/2)',
                            y='ih/2-(ih/zoom/2)',
                            d=total_frames,
                            s=f'{target_width}x{target_height}')
                    .output(output_path, vcodec='libx264', pix_fmt='yuv420p',
                            r=framerate, t=duration)
                    .overwrite_output()
                    .run(quiet=True)
                )

            elif effect_type == 2:
                # 左到右移动
                pan_width = int(target_width * 1.3)
                (
                    ffmpeg
                    .input(image_path, loop=1, t=duration, framerate=framerate)
                    .filter('scale', pan_width, target_height)
                    .filter('crop', target_width, target_height,
                            f'(iw-ow)*t/{duration}', '(ih-oh)/2')
                    .output(output_path, vcodec='libx264', pix_fmt='yuv420p',
                            r=framerate, t=duration)
                    .overwrite_output()
                    .run(quiet=True)
                )

            elif effect_type == 3:
                # 上到下移动
                pan_height = int(target_height * 1.3)
                (
                    ffmpeg
                    .input(image_path, loop=1, t=duration, framerate=framerate)
                    .filter('scale', target_width, pan_height)
                    .filter('crop', target_width, target_height,
                            '(iw-ow)/2', f'(ih-oh)*t/{duration}')
                    .output(output_path, vcodec='libx264', pix_fmt='yuv420p',
                            r=framerate, t=duration)
                    .overwrite_output()
                    .run(quiet=True)
                )

            elif effect_type == 4:
                # 下到上移动
                pan_height = int(target_height * 1.3)
                (
                    ffmpeg
                    .input(image_path, loop=1, t=duration, framerate=framerate)
                    .filter('scale', target_width, pan_height)
                    .filter('crop', target_width, target_height,
                            '(iw-ow)/2', f'(ih-oh)*(1-t/{duration})')
                    .output(output_path, vcodec='libx264', pix_fmt='yuv420p',
                            r=framerate, t=duration)
                    .overwrite_output()
                    .run(quiet=True)
                )

            elif effect_type == 5:
                # 右到左移动
                pan_width = int(target_width * 1.3)
                (
                    ffmpeg
                    .input(image_path, loop=1, t=duration, framerate=framerate)
                    .filter('scale', pan_width, target_height)
                    .filter('crop', target_width, target_height,
                            f'(iw-ow)*(1-t/{duration})', '(ih-oh)/2')
                    .output(output_path, vcodec='libx264', pix_fmt='yuv420p',
                            r=framerate, t=duration)
                    .overwrite_output()
                    .run(quiet=True)
                )

            logger.info(f"动态效果视频生成成功: {output_path}, 效果类型: {effect_type}, 时长: {duration:.2f}秒")

        except Exception as e:
            logger.error(f"创建动态效果失败: {str(e)}")
            # 如果动态效果失败，创建静态视频
            self.create_static_video(image_path, output_path, duration, target_width, target_height)

    def create_static_video(self, image_path: str, output_path: str, duration: float,
                            target_width: int, target_height: int):
        """创建静态图片视频（备用方案）"""
        try:
            framerate = 25
            (
                ffmpeg
                .input(image_path, loop=1, t=duration, framerate=framerate)
                .filter('scale', target_width, target_height)
                .output(output_path, vcodec='libx264', pix_fmt='yuv420p',
                        r=framerate, t=duration)
                .overwrite_output()
                .run(quiet=True)
            )
            logger.info(f"静态视频生成成功: {output_path}, 时长: {duration:.2f}秒")
        except Exception as e:
            logger.error(f"创建静态视频失败: {str(e)}")
            raise

    def create_subtitle_file(self, subtitles_data: List[dict], output_path: str):
        """
        创建SRT字幕文件
        subtitles_data: [{"text": "字幕文本", "start": 开始时间(秒), "end": 结束时间(秒)}]
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, subtitle in enumerate(subtitles_data, 1):
                    start_time = self.seconds_to_srt_time(subtitle['start'])
                    end_time = self.seconds_to_srt_time(subtitle['end'])

                    f.write(f"{i}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{subtitle['text']}\n\n")

            logger.info(f"字幕文件创建成功: {output_path}")
        except Exception as e:
            logger.error(f"创建字幕文件失败: {str(e)}")
            raise

    def seconds_to_srt_time(self, seconds: float) -> str:
        """将秒数转换为SRT时间格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    def extract_scene_number(self, filename: str) -> int:
        """从文件名中提取场景编号"""
        # 匹配scene_数字.png或scene_数字.jpg格式
        match = re.search(r'scene_(\d+)', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # 匹配其他可能的数字格式
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))

        return 0

    def sort_images_by_scene_number(self, image_paths: List[str]) -> List[str]:
        """根据场景编号对图片路径进行排序"""

        def get_sort_key(path):
            filename = os.path.basename(path)
            return self.extract_scene_number(filename)

        sorted_paths = sorted(image_paths, key=get_sort_key)

        # 打印排序结果用于调试
        logger.info("图片排序结果:")
        for i, path in enumerate(sorted_paths):
            filename = os.path.basename(path)
            scene_num = self.extract_scene_number(filename)
            logger.info(f"  场景 {i + 1}: {filename} (提取的编号: {scene_num})")

        return sorted_paths

    def create_video_from_images_with_audio(self, image_paths: List[str], scene_descriptions: List[str],
                                            output_path: str, session_id: str):
        """
        从图片和场景描述创建带语音和字幕的视频
        """
        try:
            if not image_paths or not scene_descriptions:
                raise ValueError("图片列表和场景描述不能为空")

            # 对图片路径按场景编号排序
            sorted_image_paths = self.sort_images_by_scene_number(image_paths)

            if len(sorted_image_paths) != len(scene_descriptions):
                logger.warning(f"图片数量({len(sorted_image_paths)})和场景描述数量({len(scene_descriptions)})不匹配")
                # 取最小长度确保一一对应
                min_length = min(len(sorted_image_paths), len(scene_descriptions))
                sorted_image_paths = sorted_image_paths[:min_length]
                scene_descriptions = scene_descriptions[:min_length]
                logger.info(f"已调整为匹配数量: {min_length}")

            # 获取第一张图片的尺寸作为目标输出尺寸
            target_width, target_height = self.get_image_dimensions(sorted_image_paths[0])
            logger.info(f"使用目标尺寸: {target_width}x{target_height}")

            # 创建临时目录
            temp_audio_dir = os.path.join("tempv", session_id)
            temp_video_dir = os.path.join("tempv", f"{session_id}_videos")
            temp_subtitle_dir = os.path.join("tempv", f"{session_id}_subtitles")

            for dir_path in [temp_audio_dir, temp_video_dir, temp_subtitle_dir, "tempv"]:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

            video_clips = []
            audio_clips = []
            all_subtitles = []
            current_time = 0.0

            # 为每个场景生成语音和动态视频
            for i, (image_path, description) in enumerate(zip(sorted_image_paths, scene_descriptions)):
                scene_num = i + 1
                logger.info(f"处理场景 {scene_num}/{len(sorted_image_paths)}")
                logger.info(f"  图片: {os.path.basename(image_path)}")
                logger.info(f"  描述: {description[:50]}...")

                # 清理场景描述并分割字幕
                clean_description = self.clean_scene_description(description)
                scene_subtitles = self.split_text_for_subtitles(clean_description)

                # 生成语音
                audio_path = os.path.join(temp_audio_dir, f"scene_{scene_num}.wav")
                audio_duration = self.text_to_speech(description, audio_path)

                # 视频时长 = 语音时长 + 0.2秒（减少额外时间以降低延迟）
                video_duration = audio_duration + 0.2

                logger.info(f"场景 {scene_num} 音频时长: {audio_duration:.2f}秒, 视频时长: {video_duration:.2f}秒")

                # 为当前场景计算字幕时间（改进时间同步）
                if scene_subtitles:
                    # 字幕开始时间提前0.1秒，减少延迟感知
                    subtitle_start_offset = max(0, current_time - 0.1)
                    subtitle_duration_per_segment = audio_duration / len(scene_subtitles)

                    for j, subtitle_text in enumerate(scene_subtitles):
                        start_time = subtitle_start_offset + j * subtitle_duration_per_segment
                        end_time = subtitle_start_offset + (j + 1) * subtitle_duration_per_segment

                        # 确保字幕不会超出当前场景的时间范围
                        end_time = min(end_time, current_time + audio_duration)

                        all_subtitles.append({
                            "text": subtitle_text,
                            "start": start_time,
                            "end": end_time
                        })

                current_time += video_duration

                # 生成带动态效果的视频片段
                video_clip_path = os.path.join(temp_video_dir, f"scene_{scene_num}.mp4")
                self.create_image_with_effect(image_path, video_clip_path, video_duration,
                                              target_width, target_height)

                video_clips.append(video_clip_path)
                audio_clips.append(audio_path)

            # 验证生成的视频片段时长
            total_expected_duration = 0
            for i, video_clip in enumerate(video_clips):
                try:
                    probe = ffmpeg.probe(video_clip)
                    actual_duration = float(probe['streams'][0]['duration'])
                    total_expected_duration += actual_duration
                    logger.info(f"视频片段 {i + 1} 实际时长: {actual_duration:.2f}秒")
                except Exception as e:
                    logger.warning(f"无法获取视频片段 {i + 1} 的时长: {str(e)}")

            logger.info(f"预计总视频时长: {total_expected_duration:.2f}秒 ({total_expected_duration / 60:.2f}分钟)")

            # 创建字幕文件
            subtitle_path = os.path.join(temp_subtitle_dir, "subtitles.srt")
            self.create_subtitle_file(all_subtitles, subtitle_path)

            # 合并所有视频片段
            logger.info("开始合并视频片段...")

            # 创建输入文件列表文件
            concat_file_path = os.path.join(temp_video_dir, "concat_list.txt")
            with open(concat_file_path, 'w', encoding='utf-8') as f:
                for video_clip in video_clips:
                    # 使用绝对路径并转换路径分隔符
                    abs_path = os.path.abspath(video_clip).replace('\\', '/')
                    f.write(f"file '{abs_path}'\n")

            # 合并视频
            temp_video_output = os.path.join(temp_video_dir, "merged_video.mp4")
            (
                ffmpeg
                .input(concat_file_path, format='concat', safe=0)
                .output(temp_video_output, c='copy')
                .overwrite_output()
                .run(quiet=True)
            )

            # 合并音频
            temp_audio_output = os.path.join(temp_audio_dir, "merged_audio.wav")
            audio_inputs = [ffmpeg.input(audio_clip) for audio_clip in audio_clips]

            if len(audio_inputs) == 1:
                final_audio = audio_inputs[0]
            else:
                final_audio = ffmpeg.concat(*audio_inputs, v=0, a=1)

            (
                final_audio
                .output(temp_audio_output)
                .overwrite_output()
                .run(quiet=True)
            )

            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # 检查BGM文件是否存在
            bgm_path = "bgm.mp3"
            temp_output_path = output_path.replace('.mp4', '_temp.mp4')

            # 将合并的视频和音频结合，并添加字幕
            video_input = ffmpeg.input(temp_video_output)
            audio_input = ffmpeg.input(temp_audio_output)

            # 添加字幕到视频（调整字幕样式以减少视觉延迟）
            video_with_subtitles = video_input.filter('subtitles', subtitle_path,
                                                      force_style='FontSize=24,PrimaryColour=&Hffffff,BackColour=&H80000000,BorderStyle=3,MarginV=50')

            if os.path.exists(bgm_path):
                logger.info("发现BGM文件，将添加背景音乐")

                # 先生成带字幕的视频（不含BGM）
                (
                    ffmpeg
                    .output(video_with_subtitles, audio_input, temp_output_path,
                            vcodec='libx264', acodec='aac', pix_fmt='yuv420p')
                    .overwrite_output()
                    .run(quiet=True)
                )

                # 获取视频总时长
                probe = ffmpeg.probe(temp_output_path)
                video_duration_total = float(probe['streams'][0]['duration'])
                logger.info(f"最终视频总时长: {video_duration_total:.2f}秒 ({video_duration_total / 60:.2f}分钟)")

                # 添加BGM
                final_video_input = ffmpeg.input(temp_output_path)
                bgm_input = ffmpeg.input(bgm_path, stream_loop=-1, t=video_duration_total)

                # 混合原音频和BGM（BGM音量设为0.3）
                mixed_audio = ffmpeg.filter([final_video_input.audio, bgm_input], 'amix',
                                            inputs=2, duration='first', weights='1.0 0.3')

                (
                    ffmpeg
                    .output(final_video_input.video, mixed_audio, output_path,
                            vcodec='libx264', acodec='aac', pix_fmt='yuv420p')
                    .overwrite_output()
                    .run(quiet=True)
                )

                # 删除临时文件
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)

            else:
                logger.info("未找到BGM文件，生成无背景音乐的视频")
                # 直接生成带字幕的视频
                (
                    ffmpeg
                    .output(video_with_subtitles, audio_input, output_path,
                            vcodec='libx264', acodec='aac', pix_fmt='yuv420p')
                    .overwrite_output()
                    .run(quiet=True)
                )

            # 验证最终输出视频的时长
            try:
                final_probe = ffmpeg.probe(output_path)
                final_duration = float(final_probe['streams'][0]['duration'])
                logger.info(f"最终输出视频时长: {final_duration:.2f}秒 ({final_duration / 60:.2f}分钟)")
            except Exception as e:
                logger.warning(f"无法获取最终视频时长: {str(e)}")

            logger.info(f"视频创建成功: {output_path}")

            # 清理临时文件
            import shutil
            for temp_dir in [temp_audio_dir, temp_video_dir, temp_subtitle_dir]:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

            return output_path

        except Exception as e:
            logger.error(f"创建视频失败: {str(e)}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            raise