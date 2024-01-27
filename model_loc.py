
import pathlib
from .utils.download_models import downloader
checkpoints_dir='/models'

def u2net_full_pretrained() -> pathlib.Path:
    """Returns u2net pretrained model location

    Returns:
        pathlib.Path: model location
    """
    return downloader("u2net.pth")


def isnet_full_pretrained() -> pathlib.Path:
    """Returns isnet pretrained model location

    Returns:
        pathlib.Path to model location
    """
    return downloader("isnet.pth")


def isnet_carveset_pretrained() -> pathlib.Path:
    """Returns isnet pretrained model location
    ISNet model finetuned on CarveSet with DUTS-HD.
    Achieves 98% of F-Beta-Score on test set

    Returns:
        pathlib.Path to model location
    """
    return downloader("isnet-97-carveset.pth")


def basnet_pretrained() -> pathlib.Path:
    """Returns basnet pretrained model location

    Returns:
        pathlib.Path: model location
    """
    return downloader("basnet.pth")


def deeplab_pretrained() -> pathlib.Path:
    """Returns basnet pretrained model location

    Returns:
        pathlib.Path: model location
    """
    return downloader("deeplab.pth")


def fba_pretrained() -> pathlib.Path:
    """Returns basnet pretrained model location

    Returns:
        pathlib.Path: model location
    """
    return downloader("fba_matting.pth")


def tracer_b7_pretrained() -> pathlib.Path:
    """Returns TRACER with EfficientNet v1 b7 encoder pretrained model location

    Returns:
        pathlib.Path: model location
    """
    return downloader("tracer_b7.pth")


def tracer_b7_carveset_finetuned() -> pathlib.Path:
    """Returns TRACER with EfficientNet v1 b7 encoder pretrained model location
    The model of tracer b7, which has been finetuned on the CarveSet dataset, with DUTS-HD subset.
    This model achieves an average F-Beta score of 96.2% on the test set.

    Returns:
        pathlib.Path to model location
    """
    return downloader("tracer-b7-carveset-finetuned.pth")


def scene_classifier_pretrained() -> pathlib.Path:
    """Returns scene classifier pretrained model location
    This model is used to classify scenes into 3 categories: hard, soft, digital

    hard - scenes with hard edges, such as objects, buildings, etc.
    soft - scenes with soft edges, such as portraits, hairs, animal, etc.
    digital - digital scenes, such as screenshots, graphics, etc.

    more info: https://huggingface.co/Carve/scene_classifier

    Returns:
        pathlib.Path: model location
    """
    return downloader("scene_classifier.pth")


def yolov4_coco_pretrained() -> pathlib.Path:
    """Returns yolov4 classifier pretrained model location
    This model is used to classify objects in images.

    Training dataset: COCO 2017
    Training classes: 80

    It's a modified version of the original model from https://github.com/Tianxiaomo/pytorch-YOLOv4 (pytorch)
    We have only added coco classnames to the model.

    Returns:
        pathlib.Path to model location
    """
    return downloader("yolov4_coco_with_classes.pth")


def cascadepsp_pretrained() -> pathlib.Path:
    """Returns cascade psp pretrained model location
    This model is used to refine segmentation masks.

    Training dataset: MSRA-10K, DUT-OMRON, ECSSD and FSS-1000
    more info: https://huggingface.co/Carve/cascadepsp

    Returns:
        pathlib.Path to model location
    """
    return downloader("cascadepsp.pth")


def cascadepsp_finetuned() -> pathlib.Path:
    """Returns cascade psp pretrained model location
    This model is used to refine segmentation masks.

    Training dataset: CarveSet with DUTS-HD, DIS.
    more info: https://huggingface.co/Carve/cascadepsp

    Returns:
        pathlib.Path to model location
    """
    return downloader("cascadepsp_finetuned_carveset.pth")


def download_all():
    u2net_full_pretrained()
    isnet_full_pretrained()
    isnet_carveset_pretrained()
    basnet_pretrained()
    deeplab_pretrained()
    fba_pretrained()
    tracer_b7_pretrained()
    tracer_b7_carveset_finetuned()
    scene_classifier_pretrained()
    yolov4_coco_pretrained()
    cascadepsp_pretrained()
    cascadepsp_finetuned()