from pathlib import Path
from pydantic import BaseModel, validator
from typing import List


class Config(BaseModel):
    # training config
    batch_size = int
    point_batches = int
    lr = float
    lr_decay = float
    data_path = Path
    model_name = str
    total_epoch = int

    # telegram notifications config
    chat_id = int
    token = str

    @validator('data_path')
    def data_path_must_be_directory(cls, v: Path):
        assert v.is_dir()
        return v
