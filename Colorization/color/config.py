from pathlib import Path
from pydantic import BaseModel, validator


class Config(BaseModel):
    batch_size: int
    point_batches: int
    lr: float
    lr_decay: float
    data_path: Path
    total_epoch: int

    @validator('data_path')
    def data_path_must_be_directory(cls, v: Path):
        assert v.is_dir()
        return v
