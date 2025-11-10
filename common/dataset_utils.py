import shutil
from pathlib import Path
from typing import Union, Optional

import datasets
import numpy as np
from datasets import get_dataset_split_names, DatasetDict, Dataset
from flwr_datasets import FederatedDataset
from flwr_datasets import partitioner
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np

from common.configs import DatasetConfig
from common.loggers import warning, info

WARNING_RECREATE_MESSAGE = (
    f"You can recreate the dataset by setting the force_create to true.\n"
)


def _process_dataset_name(name):
    """Process dataset name to ensure it is in lowercase and without special characters."""
    return name.lower().replace(" ", "_").replace("-", "_").replace(".", "_").replace("/", "_")


def _prepare_image_array(images: np.ndarray) -> np.ndarray:
    """Ensure NPZ image arrays are 4D float32 in [0, 1]."""
    if images.ndim == 4:
        processed = images
    elif images.ndim == 3:
        processed = images[..., None]
    elif images.ndim == 2:
        side = int(np.sqrt(images.shape[1]))
        if side * side != images.shape[1]:
            raise ValueError(f"Cannot reshape images of shape {images.shape} into squares.")
        processed = images.reshape((images.shape[0], side, side, 1))
    else:
        raise ValueError(f"Unsupported image shape {images.shape}")

    processed = processed.astype(np.float32, copy=False)
    if processed.size and processed.max() > 1.0:
        processed /= 255.0
    return processed


def _npz_split_dataset(images: np.ndarray, labels: np.ndarray, val_ratio: float = 0.1) -> DatasetDict:
    """Split NPZ arrays into HF DatasetDict."""
    if images.shape[0] != labels.shape[0]:
        raise ValueError("Image and label counts differ in NPZ partition.")

    total = images.shape[0]
    split_idx = int((1 - val_ratio) * total)
    if total and split_idx == 0:
        split_idx = max(1, total - 1)

    train_imgs = images[:split_idx]
    train_labels = labels[:split_idx]
    val_imgs = images[split_idx:]
    val_labels = labels[split_idx:]

    dataset_dict = {
        "train": datasets.Dataset.from_dict({"img": train_imgs, "label": train_labels})
    }

    if val_imgs.size:
        dataset_dict["test"] = datasets.Dataset.from_dict({"img": val_imgs, "label": val_labels})
    else:
        dataset_dict["test"] = datasets.Dataset.from_dict(
            {"img": np.empty((0, *images.shape[1:])), "label": np.empty((0,), dtype=np.int64)}
        )

    return DatasetDict(dataset_dict)


def _load_npz_partition(base_path: Path, partition_id) -> Optional[DatasetDict]:
    """Load per-partition NPZ fallback."""
    candidates = [
        base_path / f"{partition_id}.npz",
        base_path / f"{partition_id}.NPZ",
    ]
    for npz_path in candidates:
        if npz_path.exists():
            with np.load(npz_path) as npz_file:
                if "x_train" not in npz_file or "y_train" not in npz_file:
                    raise ValueError(f"NPZ file {npz_path} missing x_train/y_train arrays.")
                images = _prepare_image_array(npz_file["x_train"])
                labels = npz_file["y_train"].astype(np.int64)
                return _npz_split_dataset(images, labels)
    return None


def _load_npz_server_eval(base_path: Path, dataset_name: str) -> Optional[Dataset]:
    """Load centralized NPZ test split when available."""
    candidates = [
        base_path / "server_eval.npz",
        base_path / f"test_{dataset_name}.npz",
        base_path / "test.npz",
        base_path / "test_femnist.npz",
    ]

    for npz_path in candidates:
        if npz_path.exists():
            with np.load(npz_path) as npz_file:
                if "x_test" not in npz_file or "y_test" not in npz_file:
                    raise ValueError(f"NPZ test file {npz_path} missing x_test/y_test arrays.")
                images = _prepare_image_array(npz_file["x_test"])
                labels = npz_file["y_test"].astype(np.int64)
                return datasets.Dataset.from_dict({"img": images, "label": labels})
    return None


def prepare_datasets(cfg: DatasetConfig):
    clean_name = _process_dataset_name(cfg.name)
    data_path = f"{cfg.path}/{clean_name}"

    if Path(data_path).exists():
        if cfg.force_create:
            info(f"Removing existing dataset at '{data_path}' as 'force_create' is True.")
            shutil.rmtree(data_path)
        else:
            info(f"Dataset '{cfg.name}' already exists at '{data_path}'.")
            return

    partitioner_cls = getattr(partitioner, cfg.partitioner.id)
    train_partitioner = partitioner_cls(**cfg.partitioner.kwargs)
    has_test = "test" in get_dataset_split_names(cfg.name)

    # Handle server_eval when test split doesn't exist
    if cfg.server_eval and not has_test:
        warning(f"server_eval=True but dataset '{cfg.name}' has no test split. Server evaluation will be skipped.")

    # Configure partitioners
    partitioners = {"train": train_partitioner}
    if has_test:
        partitioners["test"] = 1 if cfg.server_eval else partitioner_cls(**cfg.partitioner.kwargs)

    fds = FederatedDataset(dataset=cfg.name, partitioners=partitioners)
    if cfg.server_eval and has_test:
        info("Saving centralized test split for server evaluation.")
        fds.partitioners["test"].dataset.save_to_disk(f"{data_path}/server_eval")

    for partition_id in range(fds.partitioners["train"].num_partitions):
        train_ds = fds.partitioners["train"].load_partition(partition_id)
        if cfg.server_eval:
            dset = DatasetDict({"train": train_ds})
        else:
            if has_test:
                test_ds = fds.partitioners["test"].load_partition(partition_id)
                dset = DatasetDict({"train": train_ds, "test": test_ds})
            else:
                dset = train_ds.train_test_split(test_size=cfg.test_size)

        dset.save_to_disk(f"{data_path}/{partition_id + 1}")


def get_partition(path, name, partition_id) -> Optional[Union[Dataset, DatasetDict]]:
    name = _process_dataset_name(name)
    data_path = f"{path}/{name}/{partition_id}"
    partition_path = Path(data_path)

    if partition_path.exists():
        partition: Union[Dataset, DatasetDict] = datasets.load_from_disk(data_path)
        return partition

    npz_partition = _load_npz_partition(partition_path.parent, partition_id)
    if npz_partition:
        return npz_partition

    warning(f"Dataset '{name}' not found at '{path}' or partition {partition_id} doesn't exist")
    warning(WARNING_RECREATE_MESSAGE)
    return None


def get_client_partition(path, name, partition_id) -> DatasetDict:
    dataset_dict = get_partition(path, name, partition_id)
    assert dataset_dict, f"Dataset '{name}' not found at '{path}'.\n{WARNING_RECREATE_MESSAGE}"
    assert isinstance(dataset_dict, DatasetDict), f"Dataset '{name}' is not a DatasetDict."
    return dataset_dict

def get_server_eval_dataset(path, name) -> Optional[Dataset]:
    dataset = get_partition(path, name, "server_eval")
    if not dataset:
        base_path = Path(path) / _process_dataset_name(name)
        return _load_npz_server_eval(base_path, _process_dataset_name(name))

    assert isinstance(dataset, Dataset), f"Server eval dataset '{name}' is not a Dataset."
    return dataset

def get_dataloader(dataset: Dataset, transform, batch_size: int, **dataloader_kwargs) -> DataLoader:
    dataset = dataset.with_transform(transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, **dataloader_kwargs)
    return dataloader


def basic_img_transform(img_key):
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    def apply(batch):
        batch[img_key] = [img_transform(np.asarray(row, dtype=np.float32)) for row in batch[img_key]]
        return batch

    return apply

# @hydra.main(config_path="../static/config/dataset", config_name="default", version_base=None)
# def script_call(cfg: DatasetConfig):
#     configure_logger("default", False, None, "INFO")
#     prepare_datasets(cfg)
#
#
# if __name__ == "__main__":
#     script_call()
