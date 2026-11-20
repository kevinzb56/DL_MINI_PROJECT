"""Verify a downloaded Phase 5 archive without extracting or loading model code."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.project_layout import archive_path
import statistics
import xml.etree.ElementTree as ET
import zipfile


def audit(root):
    archive = archive_path(root, 'phase5_results.zip')
    with zipfile.ZipFile(archive_path(root, 'phase4_results.zip')) as parent:
        raw_manifest = parent.read('data/manifests/pilot_v1.json')
    manifest = json.loads(raw_manifest)
    settings = json.loads((root / 'configs/fewshot.json').read_text())
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert len(names) == len(set(names)), 'Duplicate ZIP entries'
        assert z.testzip() is None, 'ZIP integrity failed'
        for name in names:
            path = PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts, name
        read = lambda name: json.loads(z.read(name))
        protocol = read('results/protocol.json')
        results = read('results/results.json')
        assert protocol == results['protocol']
        assert protocol['settings'] == settings
        assert protocol['manifest_sha256'] == hashlib.sha256(raw_manifest).hexdigest()
        assert protocol['support_draws'] == manifest['support_draws']
        assert read('results/status.json') == {'state': 'complete', 'adaptation_runs': 30}
        for model, phase, relative in [('control', 3, 'litecrack'), ('boundary', 4, 'litecrack_ba')]:
            with zipfile.ZipFile(archive_path(root, f'phase{phase}_results.zip')) as parent:
                pointer = json.loads(parent.read(f'artifacts/phase{phase}/{relative}/checkpoints/best.json'))
                snapshot = parent.read(f'artifacts/phase{phase}/{relative}/checkpoints/' + pointer['file'])
                assert hashlib.sha256(snapshot).hexdigest() == pointer['sha256'] == protocol['parents'][model]
        cases = ET.fromstring(z.read('test_evidence/tests.xml')).findall('.//testcase')
        assert len(cases) == 5
        assert all(all(c.find(tag) is None for tag in ('failure', 'error', 'skipped')) for c in cases)
        recovery = [c for c in cases if 'test_adaptation_resume' in c.attrib['name']]
        assert len(recovery) == 4
        assert {c.attrib['name'].split('[')[-1] for c in recovery} == {
            'cpu-litecrack]', 'cpu-litecrack_ba]', 'cuda-litecrack]', 'cuda-litecrack_ba]'}
        splits = {s: {r['image_id'] for r in manifest['records'] if r['split'] == s}
                  for s in ('source_validation', 'source_holdout', 'target_query', 'target_acquisition')}
        metric_files = 0

        def check_metrics(label, metrics):
            nonlocal metric_files
            for split, aggregate in metrics.items():
                rows = read(f'results/{label}_{split}.json')
                ids = [r['image_id'] for r in rows]
                assert len(ids) == len(set(ids)) == aggregate['n'] == len(splits[split])
                assert set(ids) == splits[split]
                for key in ('dice', 'iou', 'precision', 'recall', 'boundary_f1'):
                    assert all(math.isfinite(r[key]) and 0 <= r[key] <= 1 for r in rows)
                    assert math.isclose(statistics.mean(r[key] for r in rows), aggregate[key], abs_tol=1e-7)
                metric_files += 1

        assert set(results['static']) == {'control', 'boundary'}
        for model, metrics in results['static'].items():
            check_metrics(model + '_k0', metrics)
        expected = {(model, draw, k) for model in ('control', 'boundary')
                    for draw in settings['draws'] for k in settings['budgets']}
        actual = [(r['model'], r['draw'], r['k']) for r in results['adapted']]
        assert len(actual) == len(set(actual)) == 30 and set(actual) == expected
        for row in results['adapted']:
            model, draw_id, k = row['model'], row['draw'], row['k']
            folder = f'results/{model}/draw{draw_id}_k{k}'
            config = read(folder + '/config.json')
            status = read(folder + '/status.json')
            pointer = read(folder + '/checkpoints/latest.json')
            history = read(folder + '/history.json')
            draw = next(d for d in manifest['support_draws'] if d['draw'] == draw_id)
            assert config['support_ids'] == draw['supports'][str(k)]
            assert len(set(config['support_ids'])) == k
            assert set(config['support_ids']) <= splits['target_acquisition']
            assert not set(config['support_ids']) & splits['target_query']
            assert config['settings'] == settings and config['seed'] == draw['seed']
            assert config['parent_sha256'] == protocol['parents'][model]
            steps = settings['epochs'] * math.ceil(k / settings['batch_size'])
            assert status['state'] == 'adaptation_complete' and status['encoder_unchanged'] is True
            assert status['step'] == pointer['global_step'] == len(history) == steps
            assert status['epoch'] == pointer['epoch'] == settings['epochs']
            assert PurePosixPath(pointer['file']).name == pointer['file']
            snapshot = z.read(folder + '/checkpoints/' + pointer['file'])
            assert hashlib.sha256(snapshot).hexdigest() == pointer['sha256'] == row['checkpoint_sha256']
            check_metrics(f'{model}_draw{draw_id}_k{k}', row['results'])
            for field, split in [('target_dice_gain_points', 'target_query'), ('source_dice_change_points', 'source_holdout')]:
                change = 100 * (row['results'][split]['dice'] - results['static'][model][split]['dice'])
                assert math.isclose(change, row[field], abs_tol=1e-9)
        summary = []
        for model in ('control', 'boundary'):
            for k in (0, *settings['budgets']):
                group = [r for r in results['adapted'] if r['model'] == model and r['k'] == k]
                scores = [100*r['results']['target_query']['dice'] for r in group] if k else [100*results['static'][model]['target_query']['dice']]
                source = [r['source_dice_change_points'] for r in group]
                summary.append({'model': model, 'k': k, 'target_dice_percent_mean': statistics.mean(scores),
                                'target_dice_percent_sample_sd': statistics.stdev(scores) if k else None,
                                'source_dice_change_points_mean': statistics.mean(source) if k else 0})
    return {'archive': archive.name, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'status': 'verified', 'adaptation_runs': 30, 'checkpoint_hashes_verified': 30,
            'metric_files_verified': metric_files, 'recorded_tests_passed': len(cases), 'summary': summary,
            'limits': ['No fresh GPU inference performed locally.',
                       'Encoder equality is reported by the run and covered by recorded recovery tests; tensors were not loaded by this audit.',
                       'Five support draws share the same query set; spread is not a population confidence interval.',
                       'Existing public masks were used; human correction time was not measured.']}


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    report = audit(root)
    target = root / 'output/phase5_archive_verification.json'
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
