from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import torch

from yonokuni.models.network import YonokuniNet


@dataclass
class ModelMetadata:
    version: str
    trained_at: str
    config: Dict[str, Any]
    notes: Optional[str] = None


def save_checkpoint(
    model: YonokuniNet,
    path: str,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    metadata: Optional[ModelMetadata] = None,
) -> None:
    state = {"model": model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if metadata is not None:
        state["metadata"] = asdict(metadata)
    torch.save(state, path)


def load_checkpoint(path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
    return torch.load(path, map_location=device or torch.device("cpu"))


def export_torchscript(model: YonokuniNet, path: str, example_inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
    model.eval()
    traced = torch.jit.trace(model, example_inputs)
    traced.save(path)


def export_onnx(
    model: YonokuniNet,
    path: str,
    example_inputs: tuple[torch.Tensor, torch.Tensor],
    *,
    opset_version: int = 17,
) -> None:
    model.eval()
    torch.onnx.export(
        model,
        example_inputs,
        path,
        input_names=["board", "aux"],
        output_names=["policy_logits", "value"],
        dynamic_axes={"board": {0: "batch"}, "aux": {0: "batch"}},
        opset_version=opset_version,
    )
