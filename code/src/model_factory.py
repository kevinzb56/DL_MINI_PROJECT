"""Explicit model selection shared by training and profiling."""
from src.models import SegmentationUNet


def build_model(config):
    if config['architecture'] in ('litecrack','litecrack_ba'):
        from src.litecrack import LiteCrack
        return LiteCrack(config.get('encoder_width_mult',.5),config['architecture']=='litecrack_ba')
    return SegmentationUNet(config['architecture'],config['base_channels'],config.get('normalization','none'))
