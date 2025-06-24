# 小说转漫画服务 (本地Diffusion版本)

## 功能特点
- 🎨 使用本地Stable Diffusion模型生成图片
- 📖 支持中文小说文本输入
- 🖼️ 自动分割场景生成漫画
- 🌐 提供Web界面交互
- ⚡ 实时流式生成

## 系统要求
- **Python**: 3.8+
- **GPU**: CUDA支持(推荐)
- **内存**: 8GB+(推荐16GB)
- **存储**: 10GB+可用空间

## 安装步骤
1. 克隆项目
```bash
git clone https://github.com/HenryPotter0546/cs_group_project.git
cd cs_group_project
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 配置环境
```bash
cp .env.example .env
# 编辑.env文件配置API密钥
```

4. 启动服务
```bash
python main.py
```
访问: http://localhost:8000

## Ngrok公网访问
1. 下载客户端
```bash
# Linux/macOS
curl -O https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xzf ngrok-*.tgz

# Windows: https://ngrok.com/download 官网下载
```

2. 认证配置
```bash
./ngrok config add-authtoken YOUR_TOKEN
```

3. 启动转发
```bash
./ngrok http 8000 --host-header="localhost:8000"
```

4. 获取公网URL
```bash
curl http://localhost:4040/api/tunnels | jq
```
5. 测试环境说明

我们使用了两台电脑进行测试：

服务器电脑

- 负责运行AI模型生成漫画

转发电脑

- 负责把内网服务转发到公网
- 安装并运行Ngrok：

注意事项

- 两台电脑需要在同一个局域网内
- 服务器电脑需要一直保持运行
- Ngrok免费版每2小时需要重新启动
- 建议使用网线连接，WiFi可能不稳定

## 模型配置
编辑`model.yaml`:
```yaml
Unstable:
  path: "/path/to/model"
  type: "sdxl"
```

## 使用说明
1. 访问本地/公网URL
2. 输入小说文本
3. 设置场景数(1-20)
4. 选择模型(unstable最佳)
5. 点击生成

## 技术架构
| 组件 | 技术 |
|-------|------|
| 后端 | FastAPI |
| AI模型 | Stable Diffusion |
| 前端 | HTML+JS(SSE) |
| 公网 | Ngrok隧道 |

## 注意事项
⚠️ **重要提示**:
- 首次运行需下载模型(耗时)
- GPU显著提升生成速度
- Ngrok免费版2小时限制
- 商用需授权

## 故障排查
| 问题 | 解决方案 |
|------|----------|
| CUDA内存不足 | 减小batch_size |
| 模型下载失败 | 检查网络/手动下载 |
| Ngrok连接失败 | 检查防火墙/令牌 |

## 贡献者
- Weiheng Li
- Sheng Wang  
- Jingwei Zeng
- Xiaoyu Zhuang
- Yihao Wang

## 许可证
MIT License  
商用联系: henrypotterheng@gmail.com
```