from typing import Literal
from pathlib import Path

import tyro

import torch
from transformers import AutoModel
import numpy as np
import onnxruntime as ort

ModelType = Literal[
    "vits16",
    "vits16plus",
    "vitb16",
    "vitl16",
]


def convert(model: ModelType = "vits16", export_dir: Path = Path("../../SynthesiaPerception/exports")):
    # Load model and processor
    model_name = f"facebook/dinov3-{model}-pretrain-lvd1689m"
    dinov3 = AutoModel.from_pretrained(model_name)
    dinov3.eval()

    # Dummy input
    b = 2
    example_input = torch.randn(b, 3, 224, 224)

    # Export to ONNX
    export_dir.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        dinov3,
        (example_input,),
        export_dir / f"dinov3-{model}.onnx",
        input_names=["pixel_values"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "last_hidden_state": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Test exported model

    # Load ONNX model
    session = ort.InferenceSession(
        export_dir / f"dinov3-{model}.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    # Test inference
    example_input = torch.randn(4, 3, 224, 224)
    onnx_outputs = session.run(None, {"pixel_values": example_input.numpy()})

    for i, output in enumerate(onnx_outputs):
        print(f"ONNX output {i}:", output.shape)

    # Compare with PyTorch
    with torch.no_grad():
        pytorch_outputs = dinov3(example_input)

    print(
        "Difference:",
        ((pytorch_outputs.last_hidden_state.numpy() - onnx_outputs[0]) ** 2).mean(),
    )


if __name__ == "__main__":
    tyro.cli(convert)
