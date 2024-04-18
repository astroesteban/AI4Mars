import torch
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


def find_images_missing_labels(img_folder: Path, label_folder: Path) -> List[Path]:
    """Iterates over the training image folder and for each image checks if it
    has a corresponding label.

    Per the AI4Mars info.txt, every image is named the same as its label with
    the only difference being that images end in .JPG and labels end in .png.

    Args:
        img_folder (Path): The path to the images
        label_folder (Path): The path to the labels

    Returns:
        List[Path]: A list of image paths that are missing labels
    """
    images_missing_labels: List[str] = []

    for img_file in img_folder.iterdir():
        img_name = img_file.stem

        has_matching_label = False

        for label_file in label_folder.iterdir():
            if img_name in str(label_file.stem):
                has_matching_label = True
                break

        if not has_matching_label:
            images_missing_labels.append(img_file)

    return images_missing_labels
