import tempfile
from pathlib import Path
import unittest
from scripts.data.audit_kaggle_data import paired_files, filename_group, source_layout
import zipfile


class PairDiscoveryTests(unittest.TestCase):
    def test_related_crackmap_patches_share_group(self):
        self.assertEqual(filename_group('CrackMap','GOPR0107 (1)'),
                         filename_group('CrackMap','GOPR0107 (3)'))
        self.assertIsNone(filename_group('DeepCrack','123'))

    def test_nested_dataset_zip_extracted_without_losing_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            folder=root/'DeepCrack/dataset'
            folder.mkdir(parents=True)
            archive=folder/'DeepCrack.zip'
            with zipfile.ZipFile(archive,'w') as z:
                for name in ('train_img','train_lab','test_img','test_lab'):
                    z.writestr('DeepCrack/'+name+'/fixture.txt','fixture')
            self.assertTrue((source_layout(root)/'train_img').is_dir())
            self.assertTrue(archive.exists())
            self.assertEqual(source_layout(root).name,'DeepCrack')

    def test_unsafe_dataset_zip_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            folder=root/'DeepCrack/dataset'
            folder.mkdir(parents=True)
            with zipfile.ZipFile(folder/'DeepCrack.zip','w') as z:
                z.writestr('../../escaped.txt','bad')
            with self.assertRaisesRegex(ValueError,'Unsafe'):
                source_layout(root)
    def test_suffix_and_mismatched_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            images,masks=root/'images',root/'masks'
            images.mkdir(); masks.mkdir()
            (images/'road01.jpg').touch()
            (masks/'road01_mask.png').touch()
            self.assertEqual(paired_files(images,masks)[0][0],'road01')
            (images/'unlabeled.jpg').touch()
            with self.assertRaisesRegex(ValueError,'Unpaired'):
                paired_files(images,masks)

    def test_ambiguous_stems_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'road.jpg').touch()
            (root/'road.png').touch()
            with self.assertRaisesRegex(ValueError,'Ambiguous'):
                paired_files(root,root)
