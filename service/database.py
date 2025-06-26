# database.py
import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession 
from sqlalchemy.orm import declarative_base

load_dotenv() # 加载 .env 文件中的变量

# 从环境变量读取配置
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")

SQLALCHEMY_DATABASE_URL = f"mysql+aiomysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL,
    # 添加连接池设置
    pool_recycle=3600,  # 每小时回收一次连接
    pool_size=10,       # 连接池大小
    max_overflow=20     # 最大溢出连接数
)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)
Base = declarative_base()

# 数据库会话的依赖项
async def get_db():
    # 标准的异步生成器依赖项
    db = AsyncSessionLocal()
    try:
        yield db
    finally:
        await db.close()