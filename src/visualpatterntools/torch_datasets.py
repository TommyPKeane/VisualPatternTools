"""Utility Methods and Custom Classes for PyTorch Datasets

References:
    - https://lightning.ai/docs/pytorch/stable/common/evaluation_basic.html
    - https://lightning.ai/docs/pytorch/stable/common/evaluation_intermediate.html
    - https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html
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
    """String-Enum of Dataset Splitting Types (Aliases)

    The naming convention is setup as:

    ```python
    f"{split - type}_{train_percent}_{validation_percent}_{test_percent}"
    ```

    This assumes an industry-standard approach to splitting your full/source Training
    Dataset into:

        - `train`: Training Data used during each "epoch" (training iteration)
        - `validation`: Dataset for checking the settling conditions of the training
            iterations. This dataset is processed as an inference run at the end of each
            training "epoch" (training iteration). If the summary statistics on the
            inference run for the `validation` Dataset meet the thresholds of the
            stopping or settling conditions for the Training Run, then the training is
            ended. Otherwise the Training Run will continue onto a new "epoch" (training
            iteration).
        - `test`: Dataset used after Training has completed and the Algorithm is
            "frozen". This dataset is processed as a final Inference Run, and the final
            overall Summary Statistics for the Trained Algorithm are done on this
            Dataset. When you see a Trained Algorithm is reporting a certain F1 Score or
            a certain Statistical Accuracy, Statistical Sensitivity, or Statistical
            Specificity, then these statistics are likely summarizing ___only___ the
            performance on this `test` Dataset.

    To standardize the implementation of these splitting approaches, this module also
    contains helper methods, which are then mapped to these Enum aliases in the `dict`
    object: `dataset_splitters_map`.

    🤓📝: If you want to add a new data-splitting approach here, you will need to create
        an equivalent function, and then update the `dataset_splitters_map` object.
    """

    random_50_20_30: str = "random_50_20_30"
    random_70_10_20: str = "random_70_10_20"
    random_80_10_10: str = "random_80_10_10"


@dataclasses.dataclass
class TrainingDataset:
    """Utility Container Class for a Training Dataset and its standardized Splits

    See the `DatasetSplit` enum-class for more details on the meaning of the dataset
    splits.

    This helper class collects all the datasets into this container for easy access
    through the `.` operator, while also providing a consistent data type for function
    arguments and return values. This is meant to let you pass around your Training
    Datasets as single instance of this container, and if you have multiple different
    versions or types of Training Datasets you can use this class to create separate
    but consistently standardized instances.

    🤓📝: In practice, you'll use this class in standardization and helper methods for
        the initial setup and parsing of your Training Dataset and its internal splits
        to follow industry-standard best practices. Once you have an instance of this,
        you'd then likely use it to pass into one of the customized subclasses of the
        `lightning.LightningDataModule` that we provide in this package. Then during
        actual training, testing, or inference runs you would rely on the customized
        `lightning.LightningDataModule`, but you'd have this class as an intermediary
        low-level helper to keep track of where your training data came from and how it
        was split.

    References:
        - https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
        - https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html
    """

    train: torch.Dataset
    validation: torch.Dataset
    test: torch.Dataset
    split_type: DatasetSplit
    source: torch.Dataset
    source_name: str
    source_version: str

    def to_default_lightning_data_module(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        **datamodule_kwargs,
    ) -> lightning.LightningDataModule:
        """Create a `lightning.LightningDataModule` instance from the current datasets

        The created `lightning.LightningDataModule` is just the default one, so the
        dataloaders are auto-generated and relatively simplistic, with shuffling only
        enabled on the `self.train` dataset.

        If you wanted, you could use this to generate the instance, and then you could
        do a _post hoc_ override of the `torch.DataLoader` instances if you wanted to
        further customize it. If you do this, though, you may have to pass the
        `batch_size` and `num_workers` values to your customized DataLoaders again
        though.

        Args:
            batch_size (int, optional): Batch size for the default DataLoaders
            num_workers (int, optional): Number of Worker Nodes/Subprocesses to use with
                the DataLoaders. The default value of `0` means that this will be a
                single process and not distributed on its own. If you're using
                distributed training, the DataLoaders may still get distributed across
                to the different Nodes/Subprocesses per your
            datamodule_kwargs (dict, optional): Keyword arguments that will be passed to
                the `lightning.LightningDataModule`.

        Returns:
            lightning.LightningDataModule: New Data Module instance from the current
                Dataset instances.
        """
        data_module_obj = lightning.LightningDataModule.from_datasets(
            train_dataset=self.train,
            val_dataset=self.validation,
            test_dataset=self.test,
            predict_dataset=None,  # Inference
            batch_size=batch_size,
            num_workers=num_workers,
            **datamodule_kwargs,
        )

        return data_module_obj


def dataset_split_random_r_v_e(
    dataset_obj: torch.Dataset,
    train_percent: int,
    validation_percent: int,
    test_percent: int,
    dataset_name: str,
    dataset_version: str,
    torch_generator: torch.Generator = DEFAULT_TORCH_RANDOM_GENERATOR,
) -> TrainingDataset:
    """(Generic) Create a `TrainingDataset` instance with a Random Split

    Args:
        dataset_obj (torch.Dataset): Source Dataset
        train_percent (int): Percentage (out of 100%) of the Source Dataset that should
            be used as the `train` Dataset subset
        validation_percent (int): Percentage (out of 100%) of the Source Dataset that
            should be used as the `validation` Dataset subset
        test_percent (int): Percentage (out of 100%) of the Source Dataset that should
            be used as the `test` Dataset subset
        dataset_name (str): Name of the dataset for logging and versioning
        dataset_version (str): Version of the source dataset for logging and versioning
        torch_generator (torch.Generator, optional): Pseudorandom Generator for doing
            the dataset splitting. If you use a constant seed value with this, you can
            assure that the splitting will be consistent between subsequent runs of any
            scripting using this function.

    Returns:
        TrainingDataset: Container Class instance with the Training Dataset info and
            the random splits according to the given percentages.
    """
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
        source_name=dataset_name,
        source_version=dataset_version,
    )

    return training_dataset_obj


def dataset_split_random_50_20_30(
    dataset_obj: torch.Dataset,
    dataset_name: str,
    dataset_version: str,
    torch_generator: torch.Generator = DEFAULT_TORCH_RANDOM_GENERATOR,
) -> TrainingDataset:
    """Create a `TrainingDataset` instance with a 50/20/30 Random Split

    Source Dataset splits are:
        - `train`: `50%`
        - `validation`: `20%`
        - `test`: `30%`

    Args:
        dataset_obj (torch.Dataset): Source Dataset
        dataset_name (str): Name of the dataset for logging and versioning
        dataset_version (str): Version of the source dataset for logging and versioning
        torch_generator (torch.Generator, optional): Pseudorandom Generator for doing
            the dataset splitting. If you use a constant seed value with this, you can
            assure that the splitting will be consistent between subsequent runs of any
            scripting using this function.

    Returns:
        TrainingDataset: Container Class instance with the Training Dataset info and
            the random splits according to the given 50/20/30 percentage splits.
    """
    training_dataset_obj: TrainingDataset = dataset_split_random_r_v_e(
        dataset_obj=dataset_obj,
        train_percent=50,
        validation_percent=20,
        test_percent=30,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        torch_generator=torch_generator,
    )

    return training_dataset_obj


def dataset_split_random_70_10_20(
    dataset_obj: torch.Dataset,
    dataset_name: str,
    dataset_version: str,
    torch_generator: torch.Generator = DEFAULT_TORCH_RANDOM_GENERATOR,
) -> TrainingDataset:
    """Create a `TrainingDataset` instance with a 70/10/20 Random Split

    Source Dataset splits are:
        - `train`: `70%`
        - `validation`: `10%`
        - `test`: `20%`

    Args:
        dataset_obj (torch.Dataset): Source Dataset
        dataset_name (str): Name of the dataset for logging and versioning
        dataset_version (str): Version of the source dataset for logging and versioning
        torch_generator (torch.Generator, optional): Pseudorandom Generator for doing
            the dataset splitting. If you use a constant seed value with this, you can
            assure that the splitting will be consistent between subsequent runs of any
            scripting using this function.

    Returns:
        TrainingDataset: Container Class instance with the Training Dataset info and
            the random splits according to the given 70/10/20 percentage splits.
    """
    training_dataset_obj: TrainingDataset = dataset_split_random_r_v_e(
        dataset_obj=dataset_obj,
        train_percent=70,
        validation_percent=10,
        test_percent=10,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        torch_generator=torch_generator,
    )

    return training_dataset_obj


def dataset_split_random_80_10_10(
    dataset_obj: torch.Dataset,
    dataset_name: str,
    dataset_version: str,
    torch_generator: torch.Generator = DEFAULT_TORCH_RANDOM_GENERATOR,
) -> TrainingDataset:
    """Create a `TrainingDataset` instance with a 80/10/10 Random Split

    Source Dataset splits are:
        - `train`: `80%`
        - `validation`: `10%`
        - `test`: `10%`

    Args:
        dataset_obj (torch.Dataset): Source Dataset
        dataset_name (str): Name of the dataset for logging and versioning
        dataset_version (str): Version of the source dataset for logging and versioning
        torch_generator (torch.Generator, optional): Pseudorandom Generator for doing
            the dataset splitting. If you use a constant seed value with this, you can
            assure that the splitting will be consistent between subsequent runs of any
            scripting using this function.

    Returns:
        TrainingDataset: Container Class instance with the Training Dataset info and
            the random splits according to the given 80/10/10 percentage splits.
    """
    training_dataset_obj: TrainingDataset = dataset_split_random_r_v_e(
        dataset_obj=dataset_obj,
        train_percent=70,
        validation_percent=10,
        test_percent=10,
        dataset_name=dataset_name,
        dataset_version=dataset_version,
        torch_generator=torch_generator,
    )

    return training_dataset_obj


dataset_splitters_map: dict[DatasetSplit, typing.Callable] = {
    DatasetSplit.random_50_20_30: dataset_split_random_50_20_30,
    DatasetSplit.random_70_10_20: dataset_split_random_70_10_20,
    DatasetSplit.random_80_10_10: dataset_split_random_80_10_10,
}
