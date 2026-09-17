"""Evaluate classical detection against YOLO boxes, reading a ZIP without extraction.

Groups augmented .rf. variants before splitting. Labels are used only for
evaluation, never passed to the detector. No content in the archive is executed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import sys
import time
import zipfile

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plate_detection import DetectorConfig, detect_plate


def box_iou(box, truth):
    x0, y0 = max(box[0], truth[0]), max(box[1], truth[1])
    x1, y1 = min(box[2], truth[2]), min(box[3], truth[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - intersection
    return intersection / union if union > 0 else 0.0


def load_dataset(archive):
    records = []
    with zipfile.ZipFile(archive) as dataset:
        names = sorted(name for name in dataset.namelist()
                       if '/images/' in name and PurePosixPath(name).suffix.lower() in ('.jpg', '.jpeg', '.png'))
        groups = sorted({PurePosixPath(name).stem.split('.rf.')[0] for name in names})
        random.Random(42).shuffle(groups)
        first, second = int(len(groups) * .6), int(len(groups) * .8)
        split_by_group = {group: ('tune' if index < first else 'validation' if index < second else 'test')
                          for index, group in enumerate(groups)}
        for name in names:
            label_path = str(PurePosixPath(name.replace('/images/', '/labels/')).with_suffix('.txt'))
            labels = []
            for line in dataset.read(label_path).decode('utf-8').splitlines():
                values = [float(value) for value in line.split()]
                if len(values) != 5 or values[0] != 0:
                    raise ValueError(f'Expected one-class YOLO box labels: {label_path}')
                _, x, y, width, height = values
                labels.append([max(0, x - width / 2), max(0, y - height / 2),
                               min(1, x + width / 2), min(1, y + height / 2)])
            group = PurePosixPath(name).stem.split('.rf.')[0]
            records.append({'name': name, 'group': group, 'split': split_by_group[group],
                            'labels': labels, 'encoded': dataset.read(name)})
    return records


def evaluate_one(task):
    cv2.setNumThreads(1)
    record, experiment = task
    image = cv2.imdecode(np.frombuffer(record['encoded'], np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {record['name']}")
    height, width = image.shape[:2]
    ratio = min(1.0, 1600 / max(height, width))
    if ratio < 1:
        image = cv2.resize(image, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_AREA)
    height, width = image.shape[:2]
    settings = DetectorConfig(**experiment.get('scoring', {}))
    started = time.perf_counter()
    result = detect_plate(image, **experiment['preprocessing'], config=settings)
    elapsed = time.perf_counter() - started
    box = None
    if result.corners is not None:
        points = result.corners / np.array([width, height])
        box = [*points.min(axis=0).tolist(), *points.max(axis=0).tolist()]
    iou = max((box_iou(box, truth) for truth in record['labels']), default=0.0) if box is not None else 0.0
    return {'name': record['name'], 'group': record['group'], 'split': record['split'],
            'iou': iou, 'box': box, 'seconds': elapsed,
            'score': None if result.selected is None else result.selected.score,
            'corner_source': None if result.selected is None else result.selected.corner_source}


def summarize(rows):
    count = len(rows)
    returned = sum(row['box'] is not None for row in rows)
    hits = sum(row['iou'] >= .5 for row in rows)
    return {'images': count, 'groups': len({row['group'] for row in rows}),
            'hits_iou50': hits,
            'hit_rate_iou50': hits / count,
            'incorrect_returned': returned - hits,
            'returned_precision_iou50': hits / returned if returned else None,
            'hits_iou75': sum(row['iou'] >= .75 for row in rows),
            'mean_iou': sum(row['iou'] for row in rows) / count,
            'no_detection': sum(row['box'] is None for row in rows),
            'mean_seconds': sum(row['seconds'] for row in rows) / count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', default=['tune'])
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    records = load_dataset(args.archive)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = [{key: record[key] for key in ('name', 'group', 'split', 'labels')} for record in records]
    (args.output / 'split_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    report = {'archive': args.archive.name, 'archive_sha256': hashlib.sha256(args.archive.read_bytes()).hexdigest(),
              'seed': 42, 'grouping': 'filename stem before .rf.', 'baseline_scoring': asdict(DetectorConfig()),
              'runtime': {'python': sys.version.split()[0], 'opencv': cv2.__version__, 'numpy': np.__version__, 'workers': args.workers},
              'experiments': []}
    experiments = json.loads(args.plan.read_text(encoding='utf-8'))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for experiment in experiments:
            selected = [record for record in records if record['split'] in args.splits]
            rows = list(executor.map(evaluate_one, ((record, experiment) for record in selected), chunksize=8))
            metrics = {split: summarize([row for row in rows if row['split'] == split]) for split in args.splits}
            report['experiments'].append({'name': experiment['name'], 'settings': experiment, 'metrics': metrics})
            (args.output / f"{experiment['name']}_predictions.json").write_text(json.dumps(rows, indent=2), encoding='utf-8')
            (args.output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps({'name': experiment['name'], 'metrics': metrics}), flush=True)


if __name__ == '__main__':
    main()
