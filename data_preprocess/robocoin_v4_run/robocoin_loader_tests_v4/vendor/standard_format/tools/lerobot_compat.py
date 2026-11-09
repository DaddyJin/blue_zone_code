"""Instance-local missing-subtask compatibility for LeRobot 0.5.

Does not modify datasets, the installed library, or video seeking behavior.
"""
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class Unified55LeRobotDataset(LeRobotDataset):
    def load_hf_dataset(self):
        # Official feature helper treats shape [1] as scalar, while RoboCOIN
        # preserves singleton annotation lists. Infer their exact Arrow types.
        import pyarrow.parquet as pq
        from datasets import Features
        from lerobot.datasets.utils import load_nested_dataset, hf_transform_to_torch
        first = next(iter(sorted((self.root / "data").rglob("*.parquet"))))
        features = Features.from_arrow_schema(pq.read_schema(first))
        data = load_nested_dataset(self.root / "data", features=features, episodes=self.episodes)
        data.set_transform(hf_transform_to_torch)
        return data

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._standard_subtasks = self.meta.subtasks
        # Official __getitem__ otherwise treats -1 as an iloc index.
        self.meta.subtasks = None

    def __getitem__(self, index):
        item = super().__getitem__(index)
        raw = item.get('subtask_index')
        if raw is None:
            item['subtask'] = ''; item['mask_subtask'] = False
            return item
        ids = raw.reshape(-1).tolist() if hasattr(raw, 'reshape') else [int(raw)]
        texts = []; masks = []
        for value in ids:
            value = int(value)
            if value == -1:
                texts.append(''); masks.append(False)
            elif self._standard_subtasks is None or not 0 <= value < len(self._standard_subtasks):
                raise IndexError('Invalid Unified55 subtask id: ' + str(value))
            else:
                texts.append(str(self._standard_subtasks.iloc[value].name)); masks.append(True)
        item['subtask'] = texts[0] if len(texts) == 1 else texts
        item['mask_subtask'] = masks[0] if len(masks) == 1 else masks
        return item

def load_dataset(root, **kwargs):
    kwargs.setdefault('download_videos', False)
    kwargs.setdefault('video_backend', 'pyav')
    return Unified55LeRobotDataset('local/standard55', root=Path(root), **kwargs)
