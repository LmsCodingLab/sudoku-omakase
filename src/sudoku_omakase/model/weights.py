import pooch
from sudoku_omakase.model.models import ModelType

odie = pooch.create(
    pooch.os_cache("sudoku_omakase_weights"),
    base_url="https://github.com/LmsCodingLab/sudoku-omakase/releases/download/weights/",
    registry={
        "basic_cnn_model.pth": "sha256:033bc8c12b655aaa98b598ce892a1d0afca2851c0f8733281c493b0d55876565",
        "resnet_model.pth": "sha256:fe2c6c9d35d6254b2a061d99725113a1eb781d1bb06c451dff2301f1f69f57bd",
        "resnext_model.pth": "sha256:50decf440080f4e8cecef79f26aaede1651e0dd5d601a9cea6c9b3e087876101",
    }
)

def load_model_from_origin(model_type: ModelType) -> str:
    if model_type == ModelType.SMALL:
        return odie.fetch("basic_cnn_model.pth", progressbar=True)
    elif model_type == ModelType.NORMAL:
        return odie.fetch("resnet_model.pth", progressbar=True)
    elif model_type == ModelType.BIG:
        return odie.fetch("resnext_model.pth", progressbar=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Expected ModelType.SMALL, ModelType.NORMAL, or ModelType.BIG.")