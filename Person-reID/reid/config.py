from pathlib import Path
from pydantic import BaseModel, validator
from typing import List


class Config(BaseModel):
    data_path: Path
    batch_size: int
    color_jitter: bool
    lr: float
    model_name: str
    total_epoch: int
    warm_epoch: int
    token: str
    chat_id: int
    scale: List[float]
    multi: bool

    @validator('data_path')
    def data_path_must_be_directory(cls, v: Path):
        assert v.is_dir()
        return v
