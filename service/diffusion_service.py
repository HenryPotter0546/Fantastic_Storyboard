import os
import time
import requests
import logging
from dotenv import load_dotenv
import aiohttp
import asyncio

load_dotenv()

logger = logging.getLogger(__name__)

class MidjourneyService:
    def __init__(self):
        self.api_url = os.getenv('MIDJOURNEY_API_URL')
        self.oref = os.getenv('MIDJOURNEY_OREF')
        self.proxy_url = os.getenv('MIDJOURNEY_PROXY_URL', self.api_url)  # 使用MIDJOURNEY_PROXY_URL或默认使用api_url
        if not self.api_url or not self.oref:
            raise ValueError("未设置 MIDJOURNEY_API_URL 或 MIDJOURNEY_OREF 环境变量")
        
        # 确保proxy_url不以斜杠结尾
        if self.proxy_url and self.proxy_url.endswith('/'):
            self.proxy_url = self.proxy_url[:-1]
            
        logger.info(f"MidjourneyService initialized with proxy_url: {self.proxy_url}")
        
    def generate_image(self, prompt):
        """生成图片并返回图片URL"""
        try:
            # 提交任务
            logger.info(f"正在提交任务，提示词: {prompt}")
            
            # 检查环境变量
            if not self.api_url or not self.oref:
                raise ValueError("MIDJOURNEY_API_URL 或 MIDJOURNEY_OREF 环境变量未设置")
            
            # 敏感词替换映射
            sensitive_word_mapping = {
                "corpse": "figure",
                "dead": "lifeless",
                "blood": "crimson",
                "bloodshot": "tired",
                "bloody": "stained",
                "gore": "darkness",
                "kill": "defeat",
                "death": "end",
                "murder": "conflict",
                "suicide": "sacrifice",
                "torture": "suffering",
                "weapon": "tool",
                "gun": "device",
                "knife": "blade",
                "bomb": "explosive",
                "drug": "substance",
                "alcohol": "beverage",
                "nude": "unclothed",
                "naked": "bare",
                "sex": "intimate",
                "porn": "adult",
                "explicit": "revealing",
                "violence": "tension",
                "violent": "intense",
                "brutal": "harsh",
                "cruel": "stern",
                "horror": "thriller",
                "scary": "eerie",
                "terrifying": "intense",
                "frightening": "dramatic",
                "gruesome": "dark",
                "grisly": "somber",
                "mutilated": "altered",
                "dismembered": "separated",
                "decapitated": "severed",
                "slaughter": "conflict",
                "massacre": "battle",
                "carnage": "chaos",
                "guts": "innards",
                "flesh": "tissue",
                "wound": "injury",
                "scar": "mark",
                "bleeding": "flowing",
                "hemorrhage": "flow",
                "trauma": "injury",
                "pain": "discomfort",
                "suffering": "struggle",
                "agony": "distress",
                "torment": "trial",
                "torture": "ordeal",
                "abuse": "mistreatment",
                "victim": "target",
                "perpetrator": "actor",
                "criminal": "offender",
                "murderer": "culprit",
                "killer": "perpetrator",
                "assassin": "agent",
                "executioner": "operator",
                "butcher": "processor",
                "slayer": "defeater",
                "destroyer": "eliminator",
                "annihilator": "remover",
                "exterminator": "remover",
                "eliminator": "remover",
                "terminator": "ender",
                "destroyer": "remover",
                "annihilator": "remover",
                "exterminator": "remover",
                "eliminator": "remover",
                "terminator": "ender",
                "wang": "person",
                "Wang": "person",
                "zhang": "person",
                "Zhang": "person",
                "li": "person",
                "Li": "person",
                "liu": "person",
                "Liu": "person",
                "chen": "person",
                "Chen": "person",
                "yang": "person",
                "Yang": "person",
                "wu": "person",
                "Wu": "person",
                "xu": "person",
                "Xu": "person",
                "sun": "person",
                "Sun": "person"
            }
            
            # 构建请求数据
            request_data = {
                "prompt": prompt,
                "oref": self.oref,
                "version": 7.0
            }
            
            # 提交任务，最多重试3次
            max_submit_retries = 3
            submit_retry_count = 0
            task_id = None
            original_prompt = prompt
            
            while submit_retry_count < max_submit_retries:
                try:
                    response = requests.post(
                        f"{self.api_url}/mj/submit/imagine",
                        json=request_data,
                        timeout=30  # 设置超时时间
                    )
                    
                    # 记录原始响应
                    logger.debug(f"提交任务响应状态码: {response.status_code}")
                    logger.debug(f"提交任务响应内容: {response.text}")
                    
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            if 'result' in result and result['result']:
                                task_id = result['result']
                                break
                            else:
                                # 检查是否包含敏感词
                                if 'properties' in result and 'bannedWord' in result['properties']:
                                    banned_word = result['properties']['bannedWord']
                                    if banned_word in sensitive_word_mapping:
                                        # 替换敏感词
                                        new_prompt = original_prompt.replace(banned_word, sensitive_word_mapping[banned_word])
                                        logger.info(f"检测到敏感词 '{banned_word}'，已替换为 '{sensitive_word_mapping[banned_word]}'")
                                        request_data['prompt'] = new_prompt
                                    else:
                                        # 未知敏感词，默认替换为 person
                                        new_prompt = original_prompt.replace(banned_word, "person")
                                        logger.info(f"检测到未知敏感词 '{banned_word}'，已默认替换为 'person'")
                                        request_data['prompt'] = new_prompt
                                logger.warning(f"任务ID未返回，响应内容: {response.text}")
                        except ValueError as e:
                            logger.error(f"解析响应JSON失败: {str(e)}")
                    else:
                        logger.warning(f"提交任务失败，状态码: {response.status_code}, 响应: {response.text}")
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"请求异常: {str(e)}")
                
                submit_retry_count += 1
                if submit_retry_count < max_submit_retries:
                    logger.info(f"重试提交任务，第 {submit_retry_count + 1} 次尝试")
                    time.sleep(2)  # 等待2秒后重试
            
            if not task_id:
                raise Exception("无法获取任务ID，请检查API配置和网络连接")
            
            logger.info(f"任务提交成功，任务ID: {task_id}")
            
            # 等待任务完成
            max_retries = 30  # 最多等待60秒
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    logger.info(f"正在检查任务状态，第 {retry_count + 1} 次尝试")
                    response = requests.get(
                        f"{self.api_url}/mj/task/{task_id}/fetch",
                        timeout=30  # 设置超时时间
                    )
                    
                    # 记录原始响应
                    logger.debug(f"获取任务状态响应状态码: {response.status_code}")
                    logger.debug(f"获取任务状态响应内容: {response.text}")
                    
                    if response.status_code == 200:
                        try:
                            task_data = response.json()
                            if task_data['status'] == 'SUCCESS':
                                logger.info("任务完成，返回图片URL")
                                return task_data['imageUrl']
                            elif task_data['status'] == 'FAILURE':
                                error_msg = task_data.get('failReason', '未知错误')
                                logger.error(f"任务失败: {error_msg}")
                                raise Exception(f"任务失败: {error_msg}")
                            else:
                                logger.info(f"任务进行中，状态: {task_data.get('status', 'unknown')}")
                        except ValueError as e:
                            logger.error(f"解析任务状态失败: {str(e)}")
                            logger.error(f"响应内容: {response.text}")
                    else:
                        logger.warning(f"获取任务状态失败，状态码: {response.status_code}, 响应: {response.text}")
                    
                except requests.exceptions.RequestException as e:
                    logger.error(f"请求异常: {str(e)}")
                
                time.sleep(2)  # 等待2秒后再次查询
                retry_count += 1
            
            raise Exception("任务超时")
            
        except Exception as e:
            logger.error(f"生成图片时发生错误: {str(e)}")
            raise 

    async def handle_image_action(self, task_id: str, action: str, index: int) -> dict:
        """
        处理图片操作（放大或变体）
        
        Args:
            task_id: 任务ID
            action: 操作类型 ('UPSCALE' 或 'VARIATION')
            index: 图片索引 (1-4)
            
        Returns:
            dict: 包含操作结果的字典
        """
        try:
            logger.info(f"开始处理图片操作: task_id={task_id}, action={action}, index={index}")
            
            # 构建请求数据
            data = {
                "taskId": task_id,
                "action": action,
                "index": index
            }
            
            logger.info(f"请求数据: {data}")
            logger.info(f"请求URL: {self.proxy_url}/submit/change")
            
            # 调用midjourney-proxy的API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.proxy_url}/submit/change",
                    json=data,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    logger.info(f"响应状态码: {response.status}")
                    response_text = await response.text()
                    logger.info(f"响应内容: {response_text}")
                    
                    try:
                        result = await response.json()
                    except Exception as e:
                        logger.error(f"解析JSON失败: {e}")
                        return {
                            "success": False,
                            "message": f"解析响应失败: {response_text}"
                        }
                    
                    if result.get("code") != 1:
                        error_msg = result.get("description", "操作失败")
                        logger.error(f"操作失败: {error_msg}")
                        return {
                            "success": False,
                            "message": error_msg
                        }
                    
                    # 等待新图片生成完成
                    new_task_id = result.get("result")
                    if not new_task_id:
                        logger.error("未获取到新任务ID")
                        return {
                            "success": False,
                            "message": "未获取到新任务ID"
                        }
                    
                    logger.info(f"新任务ID: {new_task_id}")
                    
                    # 轮询等待新图片生成完成
                    max_retries = 30  # 最多等待30秒
                    for i in range(max_retries):
                        logger.info(f"检查任务状态，第 {i+1} 次尝试")
                        async with session.get(f"{self.proxy_url}/task/{new_task_id}/fetch") as task_response:
                            task_result = await task_response.json()
                            logger.info(f"任务状态: {task_result}")
                            
                            if task_result.get("status") == "SUCCESS":
                                image_url = task_result.get("imageUrl")
                                logger.info(f"任务完成，图片URL: {image_url}")
                                return {
                                    "success": True,
                                    "imageUrl": image_url
                                }
                            elif task_result.get("status") == "FAILURE":
                                fail_reason = task_result.get("failReason", "图片生成失败")
                                logger.error(f"任务失败: {fail_reason}")
                                return {
                                    "success": False,
                                    "message": fail_reason
                                }
                        await asyncio.sleep(1)
                    
                    logger.error("图片生成超时")
                    return {
                        "success": False,
                        "message": "图片生成超时"
                    }
                    
        except Exception as e:
            logger.error(f"处理图片操作失败: {str(e)}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return {
                "success": False,
                "message": f"处理失败: {str(e)}"
            } 