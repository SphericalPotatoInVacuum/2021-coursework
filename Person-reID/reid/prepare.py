from pathlib import Path
from shutil import copyfile, rmtree

from loguru import logger
from tqdm import tqdm


@logger.catch
def prepare(path: Path):
    logger.info(f'Preparing data at "{path}"')
    save_path: Path = path / '../../interim'
    if save_path.exists():
        # rmtree(save_path)
        pass
    save_path.mkdir(parents=True, exist_ok=True)

    # query
    query_path: Path = path / 'query'
    query_save_path: Path = save_path / 'query'
    query_save_path.mkdir(exist_ok=True)

    logger.info('Copying query images')
    for p in tqdm(query_path.rglob('*.jpg')):
        cls_id = p.name.split('_')[0]
        dst_path: Path = query_save_path / cls_id
        dst_path.mkdir(exist_ok=True)
        copyfile(p, dst_path / p.name)

    # multi-query
    logger.info('Copying multi-query images')
    query_path = path / 'gt_bbox'
    query_save_path = save_path / 'multi-query'
    query_save_path.mkdir(exist_ok=True)

    for p in tqdm(query_path.rglob('*.jpg')):
        cls_id = p.name.split('_')[0]
        dst_path: Path = query_save_path / cls_id
        dst_path.mkdir(exist_ok=True)
        copyfile(p, dst_path / p.name)

    # -----------------------------------------
    # gallery
    logger.info('Copying gallery images')
    gallery_path = path / 'bounding_box_test'
    gallery_save_path = save_path / 'gallery'
    gallery_save_path.mkdir(exist_ok=True)

    for p in tqdm(gallery_path.rglob('*.jpg')):
        cls_id = p.name.split('_')[0]
        dst_path: Path = gallery_save_path / cls_id
        dst_path.mkdir(exist_ok=True)
        copyfile(p, dst_path / p.name)

    # ---------------------------------------
    # train
    logger.info('Preparing train images')
    train_path = path / 'bounding_box_train'
    train_save_path = save_path / 'train'
    train_save_path.mkdir(exist_ok=True)
    val_save_path = save_path / 'val'
    val_save_path.mkdir(exist_ok=True)

    for p in tqdm(train_path.rglob('*.jpg')):
        cls_id = p.name.split('_')[0]
        dst_path: Path = val_save_path / cls_id
        if dst_path.exists():
            dst_path = train_save_path / cls_id
        dst_path.mkdir(exist_ok=True)
        copyfile(p, dst_path / p.name)
