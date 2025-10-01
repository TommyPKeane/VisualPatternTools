"""Utility Methods and Custom Classes for PyTorch Datasets

References:
    - ...
"""

import logging

import lightning
import torch
import typing


module_logger = logging.getLogger(__name__)


DEFAULT_RANDOM_SEED: int = 2025
DEFAULT_TORCH_RANDOM_GENERATOR: torch.Generator = torch.Generator().manual_seed(
    DEFAULT_RANDOM_SEED
)


class DatasetSplit(enum.StrEnum):
    random_70_10_20: str = "random_70_10_20"
    random_80_10_10: str = "random_80_10_10"


@dataclasses.dataclass
class TrainingDataset:
    train: torch.Dataset
    validation: torch.Dataset
    test: torch.Dataset
    split_type: DatasetSplit
    source: torch.Dataset


def dataset_split_random_r_v_e(
    dataset_obj: torch.Dataset,
    train_percent: int,
    validation_percent: int,
    test_percent: int,
    torch_generator: torch.Generator = DEFAULT_TORCH_RANDOM_GENERATOR,
) -> TrainingDataset:
    train_dataset, validation_dataset, test_dataset = data.random_split(
        dataset_obj,
        [train_percent, validation_percent, test_percent],
        generator=torch_generator,
    )

    training_dataset_obj = TrainingDataset(
        train=train_dataset,
        validation=validation_dataset,
        test=test_dataset,
        split_type=TrainingDataset.__value_2_member_map__(
            f"random_{train_percent}_{validation_percent}_{test_percent}",
        ),
        source=dataset_obj,
    )

    return training_dataset_obj


def dataset_split_random_70_10_20(
    dataset_obj: torch.Dataset,
    torch_generator: torch.Generator = DEFAULT_TORCH_RANDOM_GENERATOR,
) -> TrainingDataset:
    training_dataset_obj: TrainingDataset = dataset_split_random_r_v_e(
        dataset_obj=dataset_obj,
        train_percent=70,
        validation_percent=10,
        test_percent=10,
        torch_generator=torch_generator,
    )

    return training_dataset_obj


def dataset_split_random_80_10_10(
    dataset_obj: torch.Dataset,
    torch_generator: torch.Generator = DEFAULT_TORCH_RANDOM_GENERATOR,
) -> TrainingDataset:
    training_dataset_obj: TrainingDataset = dataset_split_random_r_v_e(
        dataset_obj=dataset_obj,
        train_percent=70,
        validation_percent=10,
        test_percent=10,
        torch_generator=torch_generator,
    )

    return training_dataset_obj


dataset_splitters_map: dict[DatasetSplit, typing.Callable] = {
    DatasetSplit.random_70_10_20: dataset_split_random_70_10_20,
    DatasetSplit.random_80_10_10: dataset_split_random_80_10_10,
}


# TODO: finish implementation
# Reference from: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html#lightning.pytorch.core.LightningDataModule
# class MyDataModule(lightning.LightningDataModule):
#     def prepare_data(self):
#         # download, IO, etc. Useful with shared filesystems
#         # only called on 1 GPU/TPU in distributed
#         ...

#     def setup(self, stage):
#         # make assignments here (val/train/test split)
#         # called on every process in DDP
#         dataset = RandomDataset(1, 100)
#         self.train, self.val, self.test = data.random_split(
#             dataset, [80, 10, 10], generator=torch.Generator().manual_seed(42)
#         )

#     def train_dataloader(self):
#         return data.DataLoader(self.train)

#     def val_dataloader(self):
#         return data.DataLoader(self.val)

#     def test_dataloader(self):
#         return data.DataLoader(self.test)

#     def on_exception(self, exception):
#         # clean up state after the trainer faced an exception
#         ...

#     def teardown(self):
#         # clean up state after the trainer stops, delete files...
#         # called on every process in DDP
#         ...
