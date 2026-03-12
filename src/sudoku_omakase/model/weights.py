import pooch
from sudoku_omakase.model.models import ModelType

odie = pooch.create(
    pooch.os_cache("sudoku_omakase_weights"),
    base_url="https://github.com/LmsCodingLab/sudoku-omakase/releases/download/weights/",
    registry={
        "basic_cnn_model.pth": "sha256:033bc8c12b655aaa98b598ce892a1d0afca2851c0f8733281c493b0d55876565",
        "resnet_model.pth": "sha256:aae0c90eaf9ca472e75193242646d461c0b50248c10ba7d7483df5fa3ed07704",
        "resnext_model.pth": "sha256:959d028d8f3cd03a9703e18b03371c0c1592a83222fe51201a775748b2b04429",
    }
)

def load_model_from_origin(model_type: ModelType) -> str:
    if model_type == ModelType.BAD:
        return odie.fetch("basic_cnn_model.pth", progressbar=True)
    elif model_type == ModelType.NORMAL:
        return odie.fetch("resnet_model.pth", progressbar=True)
    elif model_type == ModelType.BIG:
        return odie.fetch("resnext_model.pth", progressbar=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Expected ModelType.BAD, ModelType.NORMAL, or ModelType.BIG.")