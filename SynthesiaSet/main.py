import os
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
import tyro
from PIL import Image
from tqdm import tqdm

mi_variant = os.getenv("MI_VARIANT", "scalar_rgb")
mi.set_variant(mi_variant)  # call this before other Mitsuba imports

from src.config import SynthesiaGeneration, SynthesiaVisualization, configs
from src.scene import Scene


def generate_sample(cfg: SynthesiaGeneration, str_i: str):
    np.random.seed(worker_seed := cfg.seed + int(str_i))

    with TemporaryDirectory(suffix=str_i) as temp_dir:
        temp_path = Path(temp_dir)

        # Generate random scene
        scene = Scene.random(cfg.scene)

        # Render scene
        im = mi.render(scene.to_mitsuba(temp_path), seed=worker_seed)

        # Render keypoints
        keypoints3d = scene.piano.keypoints3d
        keypoints2d, depth = scene.camera(keypoints3d)

        # Save outputs
        image_path = cfg.output_dir / f"{str_i}.png"
        kpts_path = cfg.output_dir / f"kpts_{str_i}.npy"

        mi.util.write_bitmap(str(image_path), im)
        np.save(
            kpts_path,
            {
                "keypoints3d": keypoints3d,
                "keypoints2d": keypoints2d,
                "depth": depth,
            },
        )


def generate(cfg: SynthesiaGeneration):
    n_digits = int(math.log10(cfg.n_samples)) + 1

    with ProcessPoolExecutor(max_workers=cfg.n_jobs) as executor:
        futures = []
        for i in range(cfg.n_samples):
            str_i = str(i).zfill(n_digits)
            futures.append(executor.submit(generate_sample, cfg, str_i))

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            unit="sample",
            desc="Generating samples",
        ):
            future.result()


def visualize(cfg: SynthesiaVisualization):
    n_digits = len(list(sorted(cfg.output_dir.glob("*.png")))[0].stem)
    image_path = cfg.output_dir / f"{str(cfg.idx).zfill(n_digits)}.png"
    kpts_path = cfg.output_dir / f"kpts_{str(cfg.idx).zfill(n_digits)}.npy"

    im = Image.open(str(image_path))
    keypoints2d = np.load(kpts_path, allow_pickle=True).item()["keypoints2d"]

    plt.imshow(im)
    plt.scatter(keypoints2d[:, 0], keypoints2d[:, 1], c="r", s=5)
    plt.show()


def entrypoint():
    cfg = tyro.extras.overridable_config_cli(configs)

    if isinstance(cfg, SynthesiaGeneration):
        generate(cfg)
    elif isinstance(cfg, SynthesiaVisualization):
        visualize(cfg)


if __name__ == "__main__":
    entrypoint()
