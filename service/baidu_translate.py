# service/baidu_translate.py

import os
import random
import hashlib
import httpx 
from dotenv import load_dotenv

load_dotenv()

class BaiduTranslateService:
    def __init__(self):
        # 加载配置
        self.app_id = os.getenv('BAIDU_APP_ID')
        self.secret_key = os.getenv('BAIDU_SECRET_KEY')
        if not self.app_id or not self.secret_key:
            raise ValueError("未设置 BAIDU_APP_ID 或 BAIDU_SECRET_KEY 环境变量")
            
        self.api_url = "https://api.fanyi.baidu.com/api/trans/vip/translate"
        # 创建可复用的异步客户端实例
        self.client = httpx.AsyncClient(timeout=30.0)

    def _generate_sign(self, query: str, salt: str) -> str:
        """
        根据百度翻译API的要求生成签名。
        签名规则: appid + query + salt + 密钥 的 MD5 值
        """
        sign_str = f"{self.app_id}{query}{salt}{self.secret_key}"
        # 使用 MD5 哈希算法
        md5 = hashlib.md5(sign_str.encode('utf-8'))
        return md5.hexdigest()

    async def translate_to_english(self, text: str) -> str:
        """
        使用百度翻译API将中文文本翻译成英文。
        """
        if not text.strip():
            return ""

        # 准备请求参数
        salt = str(random.randint(32768, 65536))
        query = text
        from_lang = 'zh'  # 源语言：中文
        to_lang = 'en'    # 目标语言：英文

        # 生成签名
        sign = self._generate_sign(query, salt)

        # 构建请求参数
        params = {
            'q': query,
            'from': from_lang,
            'to': to_lang,
            'appid': self.app_id,
            'salt': salt,
            'sign': sign
        }

        try:
            # 发起异步 GET 请求 (百度翻译使用 GET)
            response = await self.client.get(self.api_url, params=params)
            response.raise_for_status() # 如果状态码不是 2xx，则抛出异常
            
            result = response.json()

            # 解析响应并处理错误
            if 'error_code' in result:
                error_msg = result.get('error_msg', '未知错误')
                raise Exception(f"百度翻译API返回错误: {error_msg} (code: {result['error_code']})")
            
            if 'trans_result' not in result or not result['trans_result']:
                raise Exception("百度翻译API未返回有效的翻译结果")

            # 提取翻译后的文本
            translated_text = result['trans_result'][0]['dst']
            return translated_text.strip()
            
        except httpx.RequestError as e:
            # 处理网络层面的错误
            raise Exception(f"调用百度翻译API失败: {str(e)}")
        except (KeyError, IndexError) as e:
            # 处理JSON解析或数据结构错误
            raise Exception(f"解析百度翻译API响应失败: {str(e)}")