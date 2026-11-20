import hashlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import torch
from src.archive_io import check_member, read_sample
from src.dataset import CrackDataset, verify_files
from scripts.experiments.train_baseline import train
from scripts.data.prepare_multisource import prepare
from scripts.data.audit_multisource_splits import TABLE
from src.archive_io import file_sha256
import csv


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        image = io.BytesIO(); Image.new('RGB',(16,16),'gray').save(image,format='PNG')
        mask = io.BytesIO(); Image.fromarray(np.eye(16,dtype=np.uint8)*255).save(mask,format='PNG')
        self.row = dict(image_id='sample',split='source_train',publisher_split='train',
                        archive='01_ground_gopro.zip',image='g/Images/sample.png',mask='g/Masks/sample.png')
        with zipfile.ZipFile(self.root/self.row['archive'],'w') as archive:
            for field, data in [('image',image.getvalue()),('mask',mask.getvalue())]:
                archive.writestr(self.row[field],data)
                self.row[field+'_sha256']=hashlib.sha256(data).hexdigest()
        self.manifest=dict(status='FROZEN_MULTISOURCE_ARCHIVE1_PILOT',records=[self.row])

    def tearDown(self): self.temp.cleanup()

    def test_zip_matches_extracted_tensors(self):
        diskrow=dict(self.row); diskrow.pop('archive')
        for field in ('image','mask'):
            path=self.root/diskrow[field]; path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(read_sample(self.root,self.row,field))
        a=CrackDataset(self.manifest,self.root,'source_train',16)[0]
        b=CrackDataset(dict(records=[diskrow]),self.root,'source_train',16)[0]
        for x,y in zip(a,b): self.assertTrue(torch.equal(x,y))
        verify_files([self.row],self.root)

    def test_tamper_is_rejected(self):
        self.row['mask_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'checksum'): verify_files([self.row],self.root)

    def test_path_escape_rejected(self):
        for name in ('../x','/x','C:/x','g/../../x','g\\x'):
            with self.assertRaises(ValueError): check_member(name)
        self.row['archive']='../outside.zip'
        with self.assertRaises(ValueError): read_sample(self.root,self.row,'image')

    def test_pilot_cannot_evaluate_test(self):
        path=self.root/'manifest.json'; path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError,'no test evaluation'):
            train(dict(protocol='multisource_archive1_pilot'),path,self.root,self.root/'run')

    def test_publisher_test_cannot_enter_training(self):
        self.row['publisher_split']='test'
        path=self.root/'manifest.json'; path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError,'split mapping'):
            train(dict(protocol='multisource_archive1_pilot'),path,self.root,self.root/'run',final_evaluation=False)


class PreparationTests(unittest.TestCase):
    def fixture(self, root, defect=None):
        rows=[]
        archive_path=root/'01_ground_gopro.zip'
        with zipfile.ZipFile(archive_path,'w') as archive:
            for i, split in enumerate(('train','val','test')):
                name=f'GoPro_3ft__{i+1:06d}'
                rows.append(dict(image_id=name,group='gopro_3ft',split=split,
                                 image_width=16,image_height=16,crack_pixels=16))
                image=io.BytesIO()
                Image.new('RGB',(16,16),(0 if defect=='duplicate' else i*40,0,0)).save(image,format='PNG')
                archive.writestr(f'gopro_3ft/Images/{name}.png',image.getvalue())
                if defect=='missing' and i==0: continue
                mask=io.BytesIO()
                raw=np.eye(16,dtype=np.uint8)*255
                if defect=='nonbinary' and i==0: raw[0,0]=128
                Image.fromarray(raw).save(mask,format='PNG')
                archive.writestr(f'gopro_3ft/Masks/{name}.png',mask.getvalue())
        table=io.StringIO(); writer=csv.DictWriter(table,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
        with zipfile.ZipFile(root/'splits.zip','w') as archive: archive.writestr(TABLE,table.getvalue())
        return file_sha256(archive_path)

    def test_audit_preserves_official_assignments(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); digest=self.fixture(root)
            with patch('scripts.data.prepare_multisource.audit',return_value={}), patch.dict('scripts.data.prepare_multisource.CHUNK_SHA256',{'01_ground_gopro.zip':digest}):
                result=prepare(root,root/'manifest.json')
            self.assertEqual([r['split'] for r in result['records']],['source_train','source_validation','source_holdout'])

    def test_bad_pairs_and_cross_split_duplicates_block_manifest(self):
        for defect, message in [('missing','Incomplete pair'),('nonbinary','Nonbinary'),('duplicate','cross-split')]:
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temp:
                root=Path(temp); digest=self.fixture(root,defect)
                with patch('scripts.data.prepare_multisource.audit',return_value={}), patch.dict('scripts.data.prepare_multisource.CHUNK_SHA256',{'01_ground_gopro.zip':digest}):
                    with self.assertRaisesRegex(ValueError,message): prepare(root,root/'manifest.json')
                self.assertFalse((root/'manifest.json').exists())


if __name__=='__main__': unittest.main()
