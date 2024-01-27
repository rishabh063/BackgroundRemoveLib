
import hashlib
import os
import warnings
from abc import ABCMeta, abstractmethod, ABC
from pathlib import Path
from typing import Optional


import requests
import tqdm

requests = requests.Session()
requests.headers.update({"User-Agent": f"Carvekit/{carvekit.version}"})
checkpoints_dir='models/'
MODELS_URLS = {
    "fba_matting.pth": {
        "repository": "Carve/fba",
        "revision": "a5d3457df0fb9c88ea19ed700d409756ca2069d1",
        "filename": "fba_matting.pth",
    },
    "cascadepsp_finetuned_carveset.pth": {
        "repository": "Carve/cascadepsp",
        "revision": "575728e071af43aa4500fc52005d3e29eb2571b4",
        "filename": "cascadepsp_finetuned_carveset.pth",
    },
}
"""
All data needed to build path relative to huggingface.co for model download
"""

MODELS_CHECKSUMS = {
    "fba_matting.pth": "890906ec94c1bfd2ad08707a63e4ccb0955d7f5d25e32853950c24c78"
    "4cbad2e59be277999defc3754905d0f15aa75702cdead3cfe669ff72f08811c52971613",
    "cascadepsp_finetuned_carveset.pth": "3c0ded519d9a0daa7a6b03cf7daf2e383797809b88cafe402b70f3c212cca"
    "f0c9522964d98eca1631fa05c8b72ecf7ce97bfb0a5ace5d82c06bf36931690d30b",
}
"""
Model -> checksum dictionary
"""


def sha512_checksum_calc(file: Path) -> str:
    """
    Calculates the SHA512 hash digest of a file on fs

    Args:
        file (Path): Path to the file

    Returns:
        SHA512 hash digest of a file.
    """
    dd = hashlib.sha512()
    with file.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            dd.update(chunk)
    return dd.hexdigest()


class CachedDownloader:
    """
    Metaclass for models downloaders.
    """

    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def name(self) -> str:
        return self.__class__.__name__

    @property
    @abstractmethod
    def fallback_downloader(self) -> Optional["CachedDownloader"]:
        """
        Property MAY be overriden in subclasses.
        Used in case if subclass failed to download model. So preferred downloader SHOULD be placed higher in the hierarchy.
        Less preferred downloader SHOULD be provided by this property.
        """
        pass

    def download_model(self, file_name: str) -> Path:
        """
        Downloads model from the internet and saves it to the cache.

        Behavior:
            If model is already downloaded it will be loaded from the cache.

            If model is already downloaded, but checksum is invalid, it will be downloaded again.

            If model download failed, fallback downloader will be used.
        """
        try:
            return self.download_model_base(file_name)
        except BaseException as e:
            if self.fallback_downloader is not None:
                warnings.warn(
                    f"Failed to download model from {self.name} downloader."
                    f" Trying to download from {self.fallback_downloader.name} downloader."
                )
                return self.fallback_downloader.download_model(file_name)
            else:
                warnings.warn(
                    f"Failed to download model from {self.name} downloader."
                    f" No fallback downloader available."
                )
                raise e

    @abstractmethod
    def download_model_base(self, model_name: str) -> Path:
        """
        Download model from any source if not cached.
        Returns:
            pathlib.Path: Path to the downloaded model.
        """

    def __call__(self, model_name: str):
        return self.download_model(model_name)


class HuggingFaceCompatibleDownloader(CachedDownloader, ABC):
    """
    Downloader for models from HuggingFace Hub.
    Private models are not supported.
    """

    def __init__(
        self,
        name: str = "Huggingface.co",
        base_url: str = "https://huggingface.co",
        fb_downloader: Optional["CachedDownloader"] = None,
    ):
        self.cache_dir = checkpoints_dir
        """SHOULD be same for all instances to prevent downloading same model multiple times
        Points to ~/.cache/carvekit/checkpoints"""
        self.base_url = base_url
        """MUST be a base url with protocol and domain name to huggingface or another, compatible in terms of models downloading API source"""
        self._name = name
        self._fallback_downloader = fb_downloader

    @property
    def fallback_downloader(self) -> Optional["CachedDownloader"]:
        return self._fallback_downloader

    @property
    def name(self):
        return self._name

    def check_for_existence(self, model_name: str) -> Optional[Path]:
        """
        Checks if model is already downloaded and cached. Verifies file integrity by checksum.
        Returns:
            Optional[pathlib.Path]: Path to the cached model if cached.
        """
        if model_name not in MODELS_URLS.keys():
            raise FileNotFoundError("Unknown model!")
        path = (
            self.cache_dir
            / MODELS_URLS[model_name]["repository"].split("/")[1]
            / model_name
        )

        if not path.exists():
            return None

        if MODELS_CHECKSUMS[path.name] != sha512_checksum_calc(path):
            warnings.warn(
                f"Invalid checksum for model {path.name}. Downloading correct model!"
            )
            os.remove(path)
            return None
        return path

    def download_model_base(self, model_name: str) -> Path:
        cached_path = self.check_for_existence(model_name)
        if cached_path is not None:
            return cached_path
        else:
            cached_path = (
                self.cache_dir
                / MODELS_URLS[model_name]["repository"].split("/")[1]
                / model_name
            )
            cached_path.parent.mkdir(parents=True, exist_ok=True)
            url = MODELS_URLS[model_name]
            hugging_face_url = f"{self.base_url}/{url['repository']}/resolve/{url['revision']}/{url['filename']}"

            try:
                r = requests.get(hugging_face_url, stream=True, timeout=10)
                if r.status_code < 400:
                    with open(cached_path, "wb") as f:
                        r.raw.decode_content = True
                        for chunk in tqdm.tqdm(
                            r,
                            desc="Downloading " + cached_path.name + " model",
                            colour="blue",
                        ):
                            f.write(chunk)
                else:
                    if r.status_code == 404:
                        raise FileNotFoundError(f"Model {model_name} not found!")
                    else:
                        raise ConnectionError(
                            f"Error {r.status_code} while downloading model {model_name}!"
                        )
            except BaseException as e:
                if cached_path.exists():
                    os.remove(cached_path)
                raise ConnectionError(
                    f"Exception caught when downloading model! "
                    f"Model name: {cached_path.name}. Exception: {str(e)}."
                )
            return cached_path


fallback_downloader: CachedDownloader = HuggingFaceCompatibleDownloader()
downloader: CachedDownloader = HuggingFaceCompatibleDownloader(
    base_url="https://cdn.carve.photos",
    fb_downloader=fallback_downloader,
    name="Carve CDN",
)
downloader._fallback_downloader = fallback_downloader
