import json
import numpy as np
import pytest
from PIL import Image

from scripts.data.prepare_milano import normalize, prepare


def test_threshold_and_transparent_labels():
    raw=Image.fromarray(np.array([[0,127,128,255]],dtype=np.uint8))
    assert normalize(raw)[1].tolist()==[[0,0,255,255]]
    hidden=Image.fromarray(np.array([[[255,255,255,0]]],dtype=np.uint8),'RGBA')
    with pytest.raises(ValueError,match='transparent'): normalize(hidden)


def test_decoded_duplicate_blocks_even_with_different_files(tmp_path):
    source=tmp_path/'raw'
    for split in ('train','val','test'):
        for kind in ('images','masks'): (source/split/kind).mkdir(parents=True)
        Image.new('RGB',(16,16),'gray').save(source/split/'images/a.png',compress_level={'train':0,'val':6,'test':9}[split])
        Image.new('L',(16,16),255).save(source/split/'masks/a.png')
    with pytest.raises(ValueError,match='overlap'): prepare(source,tmp_path/'prepared')
    audit=json.loads((tmp_path/'prepared/preparation_audit.json').read_text())
    assert audit['state']=='blocked_exact_overlap'
    assert len(audit['decoded_cross_split_duplicates'])==1
