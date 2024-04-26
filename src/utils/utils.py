import torch
import shutil
from typing import Any, List
from pathlib import Path
from fastai.vision.all import *


def cleanup_gpu_cache() -> None:
    """Utility function to cleanup unused CUDA memory"""
    with torch.no_grad():
        torch.cuda.empty_cache()

    import gc

    gc.collect()


def calc_model_size(model: Any, show: bool = True) -> int:
    """Calculates the total size, in megabytes, of a
    model

    Args:
        model (Any): The PyTorch model
        show (bool, optional): Flag to print the size. Defaults to True.

    Returns:
        int: The size of the model in megabytes
    """
    param_size: int = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2

    if show:
        print("model size: {:.3f}MB".format(size_all_mb))

    return size_all_mb


@patch
@delegates(subplots)
def plot_metrics(
    self: Recorder, nrows: int = None, ncols: int = None, figsize: int = None, **kwargs
):
    """This function serves a callback to FastAI `Recorder` objects so that
    users can plot the `Learner` metrics like the loss, error rate, accuracy,
    etc.

    Args:
        self (Recorder): _description_
        nrows (int, optional): Number of rows in the figure. Defaults to None.
        ncols (int, optional): Number of columns in the figure. Defaults to None.
        figsize (int, optional): The total figure size. Defaults to None.
    """
    metrics = np.stack(self.values)
    names = self.metric_names[1:-1]

    n = len(names) - 1

    if nrows is None and ncols is None:
        nrows = int(math.sqrt(n))
        ncols = int(np.ceil(n / nrows))
    elif nrows is None:
        nrows = int(np.ceil(n / ncols))
    elif ncols is None:
        ncols = int(np.ceil(n / nrows))
    figsize = figsize or (ncols * 6, nrows * 4)
    fig, axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
    axs = [ax if i < n else ax.set_axis_off() for i, ax in enumerate(axs.flatten())][:n]
    for i, (name, ax) in enumerate(zip(names, [axs[0]] + axs)):
        ax.plot(
            metrics[:, i],
            color="#1f77b4" if i == 0 else "#ff7f0e",
            label="valid" if i > 0 else "train",
        )
        ax.set_title(name if i > 1 else "losses")
        ax.legend(loc="best")
    plt.show()


def enable_gpu_if_available() -> Any:
    """Checks if a GPU is available and if it
    is then it prints out the GPU information
    and returns the device handle

    Returns:
        Any: The torch.device handle
    """
    # Enable GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("__CUDA VERSION:", torch.backends.cudnn.version())
        print("__Number CUDA Devices:", torch.cuda.device_count())
        print("__CUDA Device Name:", torch.cuda.get_device_name(0))
        print(
            "__CUDA Device Total Memory [GB]:",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    return device


def download_ai4mars_dataset(
    gdrive_id: str = "1iJ95GwACxEiYubXrtbxZiwxKCATQ-eEb",
    output_folder: Path = Path("/workspace/data"),
    extract: bool = True,
) -> Path:
    """Downloads the AI4Mars dataset from a Google Drive instance. Note that there does
    exist a Kaggle version, however it is incomplete.

    Args:
        gdrive_id (str, optional): The GDrive file ID. Defaults to "1iJ95GwACxEiYubXrtbxZiwxKCATQ-eEb".
        output_folder (Path, optional): The folder to save the download to. Defaults to Path("/workspace/data").
        extract (bool, optional): Flag to extract the zip file. Defaults to True.

    Raises:
        ImportError: An error is raised if gdown is not installed

    Returns:
        Path: Path to the downloaded dataset
    """
    import kaggle

    if output_folder.exists():
        return Path(output_folder / output_folder.ls()[0])

    dataset_name = "yash92328/ai4mars-terrainaware-autonomous-driving-on-mars"

    # Download the dataset to a hidden folder and extract it from kaggle
    output_folder.mkdir(parents=True, exist_ok=True)
    kaggle.api.dataset_download_cli(dataset_name, path=output_folder, unzip=extract)

    # Lets append the subfolder to our path
    dataset_path = Path(output_folder / output_folder.ls()[0])

    return dataset_path

    # try:
    #     import gdown
    # except ImportError:
    #     raise ImportError("Please install `gdown` with `pip install gdown`")

    # download_zip_file: Path = output_folder / "ai4mars.zip"

    # gdown.download(id=gdrive_id, output=str(download_zip_file))

    # if download_zip_file.exists() and extract:
    #     import zipfile

    #     extracted_dataset_path = Path(output_folder / "ai4mars")

    #     with zipfile.ZipFile(download_zip_file, "r") as zip_ref:
    #         zip_ref.extractall(extracted_dataset_path)

    #     return extracted_dataset_path

    # return download_zip_file


def prepare_dataset(
    images_path: Path, mask_path_train: Path, mask_path_test: Path
) -> List[Path]:
    """_summary_

    Args:
        image_path (Path): _description_
        mask_path_train (Path): _description_
        mask_path_test (Path): _description_

    Returns:
        List[Path]: _description_
    """
    # Setup the new image paths
    images_path_train = images_path / "train"
    images_path_train.mkdir(exist_ok=True)

    images_path_test = images_path / "test"
    images_path_test.mkdir(exist_ok=True)

    images_path_unused = images_path / "unused"
    images_path_unused.mkdir(exist_ok=True)

    # Move the test images into the test folder
    for label in mask_path_test.iterdir():
        label_name = label.stem
        label_name = label_name[:-7]

        for image in images_path.iterdir():
            if image.stem == label_name:
                shutil.move(image, images_path_test / image.name)
                break

    # Move the training images that have labels to the training folder
    for label in mask_path_train.iterdir():
        for image in images_path.iterdir():
            if label.stem == image.stem:
                shutil.move(image, images_path_train / image.name)
                break

    # Move the unlabeled images to the unused folder
    for image in images_path.iterdir():
        if image.suffix == ".JPG":
            shutil.move(image, images_path_unused / image.name)

    return (images_path_train, images_path_test, images_path_unused)
