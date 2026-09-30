"""初始化 VAD 推理所需的数据和模型，后续用于保存感知、预测结果。"""

import argparse
import copy
import importlib
import os.path as osp
import sys

from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


# 保证从任意目录执行脚本时都能导入 projects.mmdet3d_plugin。
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='初始化 VAD 预测结果提取流程')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='模型权重路径')
    parser.add_argument(
        '--cfg-options', nargs='+', action=DictAction,
        help='覆盖配置项，例如 data.workers_per_gpu=2')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    # 导入配置中注册的 VAD 数据集、pipeline 和模型模块。
    if cfg.get('custom_imports'):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)
    if cfg.get('plugin', False):
        plugin_dir = cfg.get('plugin_dir', osp.dirname(args.config))
        importlib.import_module(plugin_dir.rstrip('/').replace('/', '.'))

    # 训练集也使用验证 pipeline，以逐帧、无随机增强的方式生成预测结果。
    train_cfg = copy.deepcopy(cfg.data.val)
    train_cfg.ann_file = cfg.data.train.ann_file
    train_cfg.test_mode = True
    val_cfg = copy.deepcopy(cfg.data.val)
    val_cfg.test_mode = True
    # samples_per_gpu 属于 dataloader，不能传给 Dataset 构造函数。
    train_cfg.pop('samples_per_gpu', None)
    val_cfg.pop('samples_per_gpu', None)

    train_dataset = build_dataset(train_cfg)
    val_dataset = build_dataset(val_cfg)
    loader_kwargs = dict(
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        num_gpus=1,
        dist=False,
        shuffle=False,
    )
    train_loader = build_dataloader(train_dataset, **loader_kwargs)
    val_loader = build_dataloader(val_dataset, **loader_kwargs)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', val_dataset.CLASSES)
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()

    print(f'训练集：{len(train_dataset)} 帧，{len(train_loader)} 个 batch')
    print(f'验证集：{len(val_dataset)} 帧，{len(val_loader)} 个 batch')
    print('模型与权重加载完成；当前脚本尚未执行推理。')


if __name__ == '__main__':
    main()
