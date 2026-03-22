import json
import random
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nca import NCA
from train import load_target


def load_config(path):
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def merge_dict(base, extra):
    merged = {}
    for key, value in base.items():
        if isinstance(value, dict):
            merged[key] = merge_dict(value, {})
        else:
            merged[key] = value

    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def save_json(path, blob):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=2, sort_keys=True))


def append_jsonl(path, blob):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(blob, sort_keys=True) + "\n")


def now_stamp():
    return time.strftime("%Y%m%d-%H%M%S")


def pick_device(preferred):
    if preferred not in {None, "", "auto"}:
        if preferred == "cuda" and not torch.cuda.is_available():
            raise SystemExit("cuda requested but not available")
        if preferred == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("mps requested but not available")
        return preferred

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def setup_torch(runtime_cfg, device):
    precision = runtime_cfg.get("matmul_precision", "high")
    if precision:
        torch.set_float32_matmul_precision(precision)

    if device == "cuda":
        allow_tf32 = runtime_cfg.get("allow_tf32", True)
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = runtime_cfg.get("benchmark", True)


def dtype_from_name(name):
    name = str(name).lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def model_from_config(config, device):
    model = NCA(
        channels=config["model"]["channels"],
        hidden_size=config["model"]["hidden_size"],
        fire_rate=config["model"]["fire_rate"],
    ).to(device)

    if config["train"].get("channels_last", False) and device == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if config["train"].get("compile", False) and hasattr(model, "compile"):
        try:
            model.compile()
        except Exception as exc:
            print(f"compile skipped: {exc}")

    return model


def target_tensor_from_config(path, config, device):
    return load_target(
        path,
        target_size=config["data"]["target_size"],
        grid_size=config["data"]["grid_size"],
        device=device,
    )


def list_target_paths(config):
    root = Path(".")
    paths = sorted(root.glob(config["run"]["target_glob"]))
    if not paths:
        raise SystemExit(f"no targets matched {config['run']['target_glob']}")
    return paths


def run_dir_for(config, target_path, seed):
    return (
        Path(config["run"]["output_root"])
        / config["run"]["group_name"]
        / Path(target_path).stem
        / f"seed_{int(seed):03d}"
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rgba_image_from_state(state):
    rgba = state[0, :4].detach().cpu().clamp(0.0, 1.0)
    array = (rgba.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGBA")


def upscale_image(image, scale):
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def contact_sheet(images, columns, padding=4, bg=(12, 12, 12, 255)):
    rows = (len(images) + columns - 1) // columns
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new(
        "RGBA",
        (
            columns * width + max(columns - 1, 0) * padding,
            rows * height + max(rows - 1, 0) * padding,
        ),
        bg,
    )
    for index, image in enumerate(images):
        x = (index % columns) * (width + padding)
        y = (index // columns) * (height + padding)
        canvas.paste(image, (x, y))
    return canvas


def checkpoint_rng_blob():
    blob = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        blob["cuda"] = torch.cuda.get_rng_state_all()
    return blob


def restore_rng_blob(blob):
    if not blob:
        return
    random.setstate(blob["python"])
    np.random.set_state(blob["numpy"])
    torch.random.set_rng_state(blob["torch"])
    if torch.cuda.is_available() and "cuda" in blob:
        torch.cuda.set_rng_state_all(blob["cuda"])
