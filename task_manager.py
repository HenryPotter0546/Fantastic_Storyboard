from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from service.task_queue_service import task_queue
from service import auth, datamodels
from service.database import get_db, AsyncSession
from typing import List
from typing import Optional
router = APIRouter()

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    message: str
    position: int
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    user_id: str

class TaskSubmitResponse(BaseModel):
    task_id: str
    position: int

@router.post("/submit-comic-task", response_model=TaskSubmitResponse)
async def submit_comic_task(
    text: str,
    num_scenes: int = 10,
    steps: int = 30,
    guidance: float = 7.5,
    width: int = 512,
    height: int = 512,
    db: AsyncSession = Depends(get_db),
    current_user: datamodels.User = Depends(auth.get_current_user)
):
    """提交漫画生成任务到队列"""
    # 在实际应用中，这里应该包含任务参数验证和用户积分检查
    from main import generate_scenes_wrapper  # 避免循环导入
    
    # 提交任务到队列
    task_id = await task_queue.submit_task(
        generate_scenes_wrapper,
        text=text,
        num_scenes=num_scenes,
        steps=steps,
        guidance=guidance,
        width=width,
        height=height,
        db=db,
        current_user=current_user,
        user_id=current_user.username
    )
    
    # 获取任务状态以返回队列位置
    status = await task_queue.get_task_status(task_id)
    
    return {
        "task_id": task_id,
        "position": status["position"]
    }

@router.get("/task-status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """获取任务状态"""
    status = await task_queue.get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status

@router.get("/user-tasks", response_model=List[TaskStatusResponse])
async def get_user_tasks(
    db: AsyncSession = Depends(get_db),
    current_user: datamodels.User = Depends(auth.get_current_user)
):
    """获取当前用户的所有任务"""
    tasks = await task_queue.get_user_tasks(current_user.username)
    return tasks

@router.post("/cancel-task/{task_id}")
async def cancel_task(task_id: str):
    """取消任务"""
    await task_queue.cancel_task(task_id)
    return {"message": "Task cancellation requested"}