import torch
from torch import Tensor
import torchvision.io

from torchrl.record.loggers import Logger, CSVLogger

from collections import defaultdict
from collections.abc import Sequence
from typing import Optional

from google.cloud import storage

from human_feedback_api.models import Comparison


class PreferenceExperiment:
    def __init__(self, experiment_name: str, *,video_fps: int = 30) -> None:
        self.scalars = defaultdict(lambda: [])
        self.videos_counter = defaultdict(lambda: 0)
        self.text_counter = defaultdict(lambda: 0)
        self.experiment_name = experiment_name
        self.video_fps = video_fps
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.experiment_name)


    def add_scalar(self, name: str, value: float, global_step: Optional[int] = None) -> None:
        self.scalars[name].append((global_step, value))

    def add_video(self, name: str, vid_tensor: Tensor, global_step: Optional[int] = None, **kwargs) -> None:
        if global_step is None:
            global_step = self.videos_counter[name]
            self.videos_counter[name] += 1

        if vid_tensor.shape[-3] not in (3, 1):
            raise RuntimeError(
                "expected the video tensor to be of format [T, C, H, W] but the third channel "
                f"starting from the end isn't in (1, 3) but is {vid_tensor.shape[-3]}."
            )
        
        if vid_tensor.ndim > 4:
            vid_tensor = vid_tensor.flatten(0, vid_tensor.ndim - 4)

        vid_tensor = vid_tensor.permute((0, 2, 3, 1))
        vid_tensor = vid_tensor.expand(*vid_tensor.shape[:-1], 3)

        # Upload video to GCS
        # TODO: check filename and destination_blob_name
        filename = f"{name}_{self.videos_counter[name]}.mp4"
        destination_blob_name = f"videos/{filename}"
        self._upload_to_gcs(vid_tensor, filename, destination_blob_name, **kwargs)
        self.videos_counter[name] += 1
        

    def __repr__(self) -> str:
        return f'PreferenceExperiment(experiment_name={self.experiment_name})'
        
    def _upload_to_gcs(self, vid_tensor: Tensor, filename: str, destination_blob_name: str, **kwargs) -> None:
        kwargs.setdefault("fps", self.video_fps)
        torchvision.io.write_video(filename, vid_tensor, **kwargs)

        blob = self.bucket.blob(destination_blob_name)

        blob.upload_from_filename(filename)



class PreferenceLogger(Logger):
    def __init__(self, exp_name: str, log_dir: Optional[str] = None) -> None:
        super().__init__(exp_name=exp_name, log_dir=log_dir)
        self.experiment = self._create_experiment()


    def _create_experiment(self) -> "PreferenceExperiment":
        return PreferenceExperiment(self.exp_name)

    def log_scalar(self, name, value, step = None) -> None:
        self.experiment.add_scalar(name, value, step)
    
    def log_video(self, name: str, vid_tensor: Tensor, step: int = None, **kwargs) -> None:
        if vid_tensor.dim() != 5 or vid_tensor.size(dim=2) not in {1, 3}:
            raise ValueError(
                f"Expected video tensor to have shape [(N), T, C, H, W] with C in (1, 3), got {vid_tensor.shape}"
            )
        self.experiment.add_video(name, vid_tensor, step, **kwargs)
    
    def log_hparams(self, cfg) -> None:
        pass
    
    def __repr__(self) -> str:
        return f'PreferenceLogger(exp_name={self.exp_name})'

    def log_histogram(self, name: str, data: Sequence, **kwargs):
        pass
