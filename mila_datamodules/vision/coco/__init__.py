"""Optimized datamodules for the ImageNet dataset."""
from .coco import CocoCaptionsDataModule
from .coco_bch import CocoCaptionsBchDataModule
# Benzina format for Coco is currently unsupported as the aveerage image size in
# Coco is 640x480 and Benzina currently forces a downscale to 512x512
# from .coco_benzina import CocoCaptionsBenzinaDataModule
