import os
import httpx
from dotenv import load_dotenv

import json


load_dotenv()


class DeepSeekService:
    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("未设置 DEEPSEEK_API_KEY 环境变量")

        self.api_url = "https://api.deepseek.com/v1/chat/completions"

        # 创建一个可复用的异步客户端实例
        self.client = httpx.AsyncClient(timeout=60.0)
    async def translate_to_english(self, text: str) -> str:
        """将中文文本翻译成英文"""
        prompt = f"""请将以下中文文本翻译成英文，保持描述的准确性和艺术性。
        要求：
        1. 只返回翻译后的英文文本，不要包含任何注释或解释
        2. 保持描述的简洁性和准确性
        3. 确保翻译后的文本适合作为图片生成的提示词

        中文文本：
        {text}
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的翻译，擅长将中文场景描述翻译成地道的英文。请只返回翻译后的英文文本，不要包含任何注释、解释或额外的格式。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
            "stream": False
        }

        try:
            response = await self.client.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()

            result = response.json()
            if 'choices' not in result or not result['choices']:
                raise Exception("API返回的数据格式不正确")

            translated_text = result['choices'][0]['message']['content'].strip()
            # 移除可能存在的注释和额外格式
            translated_text = translated_text.split('\n')[0].strip()
            return translated_text

        except httpx.RequestError as e:
            raise Exception(f"调用DeepSeek API翻译失败: {str(e)}")

        except (KeyError, IndexError) as e:
            raise Exception(f"解析API响应失败: {str(e)}")
        

    async def convert_to_image_prompt(self, english_text: str) -> str:
        """将英文文本转换为适合模型生成图片的语句段"""
        prompt = f"""请将以下英文文本转换为适合AI图像生成模型的提示词格式。
        要求：
        1. 按照以下格式组织提示词，每个元素之间用逗号分隔：
           - 主体描绘：描述人物的特征（如可爱、华丽、神秘等）
           - 核心主体：描述主要人物或物体
           - 主体动作：描述人物正在进行的动作
           - 场景描述：描述场景的布局和细节
           - 风格：描述艺术风格（如像素画风、极简主义、漫画风格等）
           - 光效：描述光线效果（如聚光、逆光、霓虹灯等）
           - 色彩：描述整体色调（如暖色调、粉彩色等）
           - 视角：描述拍摄角度（如近景特写、史诗广角等）
           - 质量：描述图片质量（如极为细致、UHD超高清等）

        2. 确保描述简洁明了，适合用图片表达
        3. 不要使用方括号、编号或其他特殊符号
        4. 保持英文输出
        5.将人名替换成a man,或者a woman
        6. 只返回转换后的提示词，不要包含任何解释

        示例格式：
        mysterious, elderly man in black, standing motionless, dark hospital corridor, horror comic style, backlit, high contrast, wide angle shot, ultra detailed

        英文文本：
        {english_text}
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": """你是一个专业的AI图像生成提示词专家，擅长将描述性文本转换为适合图像生成模型的结构化提示词。
                    你需要按照以下格式生成描述，每个元素之间用逗号分隔：
                    1. 主体描绘：描述人物的特征
                    2. 核心主体：描述主要人物或物体
                    3. 主体动作：描述人物正在进行的动作
                    4. 场景描述：描述场景的布局和细节
                    5. 风格：描述艺术风格
                    6. 光效：描述光线效果
                    7. 色彩：描述整体色调
                    8. 视角：描述拍摄角度
                    9. 质量：描述图片质量

                    请确保：
                    - 使用英文描述
                    - 不要使用方括号、编号或其他特殊符号
                    - 每个元素之间用逗号分隔
                    - 描述简洁明了，适合用图片表达
                    -将人名替换成a man,或者a woman
                    - 只返回转换后的提示词，不要包含任何解释"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
            "stream": False
        }

        try:
            response = await self.client.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()

            result = response.json()
            if 'choices' not in result or not result['choices']:
                raise Exception("API返回的数据格式不正确")

            image_prompt = result['choices'][0]['message']['content'].strip()
            # 移除可能存在的注释和额外格式
            image_prompt = image_prompt.split('\n')[0].strip()
            return image_prompt

        except httpx.RequestError as e:
            error_detail = ""
            try:
                error_detail = response.json()
            except:
                error_detail = response.text
            raise Exception(f"调用DeepSeek API失败: {str(e)}\n错误详情: {error_detail}")
        except (KeyError, IndexError) as e:
            raise Exception(f"解析API响应失败: {str(e)}")


    async def split_into_scenes(self, novel_text, num_scenes=10):

        """将小说文本分割成场景描述"""
        prompt = f"""作为一个为小说配图的工作者，请将以下小说文本分割成{num_scenes}个场景，每个场景用一段简短的描述概括，适合用于生成漫画分镜。

        要求：
        1. 每个场景描述要包含以下要素，用逗号分隔：
           - 主体描绘：描述人物的特征，如可爱、华丽、神秘等
           - 核心主体：描述主要人物或物体
           - 主体动作：描述人物正在进行的动作
           - 场景描述：描述场景的布局和细节
           - 风格：描述艺术风格，如像素画风、极简主义等
           - 光效：描述光线效果，如聚光、逆光、霓虹灯等
           - 色彩：描述整体色调，如暖色调、粉彩色等
           - 视角：描述拍摄角度，如近景特写、史诗广角等
           - 质量：描述图片质量，如极为细致、UHD超高清等

        2. 描述要简洁明了，适合用图片表达
        3. 每个场景用换行分隔
        4. 不要使用方括号、编号或其他特殊符号
        5. 使用英文描述，每个元素之间用逗号分隔

        示例格式：
        mysterious, elderly man in black, standing motionless, dark hospital corridor, horror comic style, backlit, high contrast, wide angle shot, ultra detailed

        小说文本：
        {novel_text}
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": """你是一个专业的漫画分镜师，擅长将小说文本转换为适合生成漫画的场景描述。
                    你需要按照以下格式生成描述，每个元素之间用逗号分隔：
                    1. 主体描绘：描述人物的特征
                    2. 核心主体：描述主要人物或物体
                    3. 主体动作：描述人物正在进行的动作
                    4. 场景描述：描述场景的布局和细节
                    5. 风格：描述艺术风格
                    6. 光效：描述光线效果
                    7. 色彩：描述整体色调
                    8. 视角：描述拍摄角度
                    9. 质量：描述图片质量

                    请确保：
                    - 使用英文描述
                    - 不要使用方括号、编号或其他特殊符号
                    - 每个元素之间用逗号分隔
                    - 描述简洁明了，适合用图片表达"""
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
            "stream": False
        }

        try:
            response = await self.client.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()

            result = response.json()
            if 'choices' not in result or not result['choices']:
                raise Exception("API返回的数据格式不正确")

            scenes = result['choices'][0]['message']['content'].split('\n')
            return [scene.strip() for scene in scenes if scene.strip()]

        except httpx.RequestError as e:

            error_detail = ""
            try:
                error_detail = response.json()
            except:
                error_detail = response.text
            raise Exception(f"调用DeepSeek API失败: {str(e)}\n错误详情: {error_detail}")
        except (KeyError, IndexError) as e:
            raise Exception(f"解析API响应失败: {str(e)}")

    async def split_into_scenes_cn(self, novel_text, num_scenes=10):
        """
        将小说文本分割成num_scenes个场景，每个场景用一段50-100字的中文完整句子描述。
        """
        prompt = f"""请将以下小说文本分割成{num_scenes}个场景，每个场景用一段50到100字的中文完整句子描述，适合用来讲述故事情节。每个场景用换行分隔。
小说文本：
{novel_text}
"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的漫画分镜师，擅长将小说文本转换为适合生成漫画的场景描述。请用中文输出，每个场景用50-100字的完整句子描述，适合用来讲述故事情节。每个场景用换行分隔。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 2000,
            "stream": False
        }
        try:
            response = await self.client.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            if 'choices' not in result or not result['choices']:
                raise Exception("API返回的数据格式不正确")
            scenes = result['choices'][0]['message']['content'].split('\n')
            return [scene.strip() for scene in scenes if scene.strip()]
        except httpx.RequestError as e:
            error_detail = ""
            try:
                error_detail = response.json()
            except:
                error_detail = response.text
            raise Exception(f"调用DeepSeek API失败: {str(e)}\n错误详情: {error_detail}")
        except (KeyError, IndexError) as e:
            raise Exception(f"解析API响应失败: {str(e)}")
        
    async def process_novel_to_scenes(self, novel_text: str, num_scenes: int = 10) -> list[dict]:
        """
        一次性将小说文本处理成场景。
        每个场景同时包含【中文描述】和【英文绘图提示词】。
        """
        prompt = f"""
        你是一个专业的漫画改编总监。你的任务是将以下小说文本，改编成 {num_scenes} 个漫画分镜。
        对于每一个分镜，你需要完成两件事：
        1.  **中文场景描述**: 用一段流畅、优美的中文句子来描述这个场景的情节和画面感，供用户阅读。
        2.  **英文绘图提示词 (Prompt)**: 根据场景描述，生成一段高质量的、逗号分隔的英文关键词组合，用于AI绘画。这个提示词需要包含核心人物、动作、表情、环境、构图和氛围。

        请严格按照下面的 JSON 格式输出你的答案。返回一个包含 {num_scenes} 个对象的 JSON 数组，不要在前后添加任何额外的解释或文本。

        **JSON 输出格式示例:**
        ```json
        [
        {{
            "chinese_description": "一个孤独的旅人，在被雨水浸湿的霓虹街道上，默默地收起他的伞，眼神中透露出一丝迷茫。",
            "english_prompt": "1 boy, solo, futuristic city, neon lights, raining, wet streets, holding a clear umbrella, looking away, detailed, cinematic lighting, cyberpunk style, masterpiece"
        }},
        {{
            "chinese_description": "在昏暗的酒馆角落，一位神秘的女士轻轻摇晃着手中的酒杯，猩红色的液体在杯中旋转，她的嘴角挂着一抹难以捉摸的微笑。",
            "english_prompt": "1 girl, solo, sitting at a table in a dark bar, holding a wine glass, red liquid, mysterious smile, dimly lit, film noir style, detailed face, atmospheric"
        }}
        ]
        ```
        小说文本：
        {novel_text}
        """

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个遵循指令的专家，会严格按照用户要求的 JSON 格式输出结果。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.6,
            "max_tokens": 3000 
        }

        raw_content = ""
        try:
            response = await self.client.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            if 'choices' not in result or not result['choices']:
                raise Exception("API返回的数据格式不正确")
                
            raw_content = result['choices'][0]['message']['content']
            
            # 更健壮的 JSON 提取方式
            json_str = raw_content
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0].strip()
            elif '```' in json_str:
                json_str = json_str.split('```')[1].strip()

            # 解析 JSON 字符串
            scenes_data = json.loads(json_str)
            
            if not isinstance(scenes_data, list):
                raise ValueError("AI 没有返回有效的 JSON 数组")

            return scenes_data

        except httpx.RequestError as e:
            error_detail = response.text if 'response' in locals() else str(e)
            raise Exception(f"调用DeepSeek API失败: {str(e)}\n错误详情: {error_detail}")
        except (json.JSONDecodeError, ValueError) as e:
            raise Exception(f"解析返回的JSON失败: {e}\n原始返回内容: {raw_content}")
        except Exception as e:
            raise Exception(f"处理过程中发生错误: {str(e)}")