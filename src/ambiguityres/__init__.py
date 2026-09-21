import pathlib
from dataclasses import dataclass

@dataclass
class ASSETS_DIR:
    ROOT = pathlib.Path(__file__).parents[1] / "assets"
    IMAGES = ROOT / "images"

    @staticmethod
    def get_dir(env="real", mode="train"):
        assert env in ["sim", "real"]
        assert mode in ["train", "test"]
        return ASSETS_DIR.DATA / env / mode

from .vlm_chat import VlmChat
from .vlm_model import AmbresStructured, ambresFewShotPrompt, ObjGrounding, LocGrounding, SceneGrounding