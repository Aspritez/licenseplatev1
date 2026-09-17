import io
import unittest
import zipfile

from scripts.evaluate_detector import box_iou, load_dataset, summarize


class DatasetEvaluationTests(unittest.TestCase):
    def test_augmented_variants_stay_in_the_same_split(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as dataset:
            for source in range(10):
                for variant in range(3):
                    stem = f'car_{source}.rf.{variant}'
                    dataset.writestr(f'train/images/{stem}.jpg', b'image payload')
                    dataset.writestr(f'train/labels/{stem}.txt', '0 0.5 0.5 0.4 0.2\n')
        archive.seek(0)
        records = load_dataset(archive)
        splits = {split: {r['group'] for r in records if r['split'] == split}
                  for split in ('tune', 'validation', 'test')}
        self.assertEqual([len(groups) for groups in splits.values()], [6, 2, 2])
        self.assertFalse(splits['tune'] & splits['validation'])
        self.assertFalse(splits['tune'] & splits['test'])
        self.assertFalse(splits['validation'] & splits['test'])
        for group in set(record['group'] for record in records):
            variants = [record for record in records if record['group'] == group]
            self.assertEqual(len(variants), 3)
            self.assertEqual(len({record['split'] for record in variants}), 1)

    def test_iou_and_missed_detection_denominator(self):
        self.assertEqual(box_iou([0, 0, 1, 1], [0, 0, 1, 1]), 1)
        self.assertEqual(box_iou([0, 0, .2, .2], [.5, .5, 1, 1]), 0)
        self.assertAlmostEqual(box_iou([0, 0, 1, 1], [0, 0, .5, 1]), .5)
        rows = [{'group': 'one', 'iou': .8, 'box': [0, 0, 1, 1], 'seconds': .1},
                {'group': 'two', 'iou': 0, 'box': None, 'seconds': .1}]
        result = summarize(rows)
        self.assertEqual(result['hit_rate_iou50'], .5)
        self.assertEqual(result['mean_iou'], .4)
        self.assertEqual(result['no_detection'], 1)


if __name__ == '__main__':
    unittest.main()
