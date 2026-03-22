import argparse
import copy
import math
import time
from contextlib import nullcontext
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from PIL import Image

from nca import make_seed
from v2.common import (
    append_jsonl,
    checkpoint_rng_blob,
    contact_sheet,
    dtype_from_name,
    load_config,
    merge_dict,
    model_from_config,
    now_stamp,
    pick_device,
    restore_rng_blob,
    rgba_image_from_state,
    run_dir_for,
    save_json,
    seed_everything,
    setup_torch,
    target_tensor_from_config,
    upscale_image,
)


def rollout_bounds(config, progress):
    train_cfg = config["train"]
    start_min = train_cfg["start_rollout_min"]
    start_max = train_cfg["start_rollout_max"]
    end_min = train_cfg["end_rollout_min"]
    end_max = train_cfg["end_rollout_max"]

    cur_min = int(round(start_min + (end_min - start_min) * progress))
    cur_max = int(round(start_max + (end_max - start_max) * progress))
    return cur_min, max(cur_min, cur_max)


def grid_cache(size, device):
    yy = torch.arange(size, device=device).view(1, 1, size, 1)
    xx = torch.arange(size, device=device).view(1, 1, 1, size)
    return yy, xx


def make_seed_batch(batch_size, config, device):
    return make_seed(
        batch_size,
        channels=config["model"]["channels"],
        height=config["data"]["grid_size"],
        width=config["data"]["grid_size"],
        device=device,
    )


def maybe_channels_last(tensor, config, device):
    if config["train"].get("channels_last", False) and device == "cuda":
        return tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def damage_batch(batch, config, yy, xx):
    train_cfg = config["train"]
    if train_cfg["damage_prob"] <= 0:
        return batch

    start = min(train_cfg["fresh_seed_count"], batch.shape[0])
    if start >= batch.shape[0]:
        return batch

    samples = batch[start:]
    count = samples.shape[0]
    size = samples.shape[-1]
    device = samples.device

    flags = torch.rand(count, device=device) < train_cfg["damage_prob"]
    if not bool(flags.any()):
        return batch

    radius_min = max(2, int(round(size * train_cfg["damage_min_radius_frac"])))
    radius_max = max(radius_min + 1, int(round(size * train_cfg["damage_max_radius_frac"])))

    radii = torch.randint(radius_min, radius_max + 1, (count,), device=device)
    span = size - radii * 2
    cx = (torch.rand(count, device=device) * span.float()).long() + radii
    cy = (torch.rand(count, device=device) * span.float()).long() + radii

    crater = (
        (xx - cx.view(count, 1, 1, 1)).pow(2)
        + (yy - cy.view(count, 1, 1, 1)).pow(2)
        <= radii.view(count, 1, 1, 1).pow(2)
    )
    crater = crater & flags.view(count, 1, 1, 1)
    samples = torch.where(crater.expand_as(samples), torch.zeros_like(samples), samples)
    batch[start:] = samples
    return batch


def build_masks(target, config):
    loss_cfg = config["loss"]
    target_alpha = target[:, 3:4]
    kernel = int(loss_cfg["overflow_kernel"])
    if kernel % 2 == 0:
        kernel += 1
    support = (F.max_pool2d(target_alpha, kernel, stride=1, padding=kernel // 2) > 0.01).float()
    fg = (target_alpha > 0.01).float()
    bg = 1.0 - fg
    overflow = 1.0 - support
    return {
        "fg": fg,
        "bg": bg,
        "overflow": overflow,
        "alpha_mean": target_alpha.mean(),
    }


def loss_terms(state, target, masks, config):
    pred = state[:, :4].clamp(0.0, 1.0)
    target_batch = target.expand(state.shape[0], -1, -1, -1)
    fg = masks["fg"].expand(state.shape[0], -1, -1, -1)
    bg = masks["bg"].expand(state.shape[0], -1, -1, -1)
    overflow = masks["overflow"].expand(state.shape[0], -1, -1, -1)

    pred_alpha = pred[:, 3:4]
    target_alpha = target_batch[:, 3:4]

    rgba = (pred - target_batch).pow(2).mean()
    rgb = ((pred[:, :3] - target_batch[:, :3]).pow(2) * target_alpha).sum()
    rgb = rgb / (target_alpha.sum() * 3.0 + 1e-6)
    alpha = (pred_alpha - target_alpha).pow(2).mean()
    bg_alpha = (pred_alpha.pow(2) * bg).sum() / (bg.sum() + 1e-6)
    overflow_alpha = (pred_alpha.pow(2) * overflow).sum() / (overflow.sum() + 1e-6)
    mass = (pred_alpha.mean(dim=(1, 2, 3)) - masks["alpha_mean"]).pow(2).mean()
    hidden = state[:, 4:].pow(2).mean()

    weights = config["loss"]
    total = (
        rgba * weights["rgba_weight"]
        + rgb * weights["rgb_weight"]
        + alpha * weights["alpha_weight"]
        + bg_alpha * weights["bg_alpha_weight"]
        + overflow_alpha * weights["overflow_weight"]
        + mass * weights["mass_weight"]
        + hidden * weights["hidden_weight"]
    )

    return {
        "total": total,
        "rgba": rgba,
        "rgb": rgb,
        "alpha": alpha,
        "bg_alpha": bg_alpha,
        "overflow_alpha": overflow_alpha,
        "mass": mass,
        "hidden": hidden,
    }


def crater_state(state, radius, yy, xx, x, y):
    mask = ((xx - x).pow(2) + (yy - y).pow(2)) <= radius * radius
    return torch.where(mask.expand_as(state), torch.zeros_like(state), state)


def eval_score(eval_points, recovery_point, config):
    tail = max(1, int(config["eval"]["best_tail"]))
    recent = eval_points[-tail:]
    recent_score = sum(point["total"] for point in recent) / len(recent)
    return recent_score * (1.0 - config["eval"]["score_recovery_weight"]) + recovery_point["total"] * config["eval"]["score_recovery_weight"]


def scalar_blob(metrics):
    return {key: float(value.detach()) if torch.is_tensor(value) else float(value) for key, value in metrics.items()}


def preview_sheet(target, rollout_frames, recovery_frames, config):
    scale = config["eval"]["preview_scale"]
    images = [upscale_image(rgba_image_from_state(target), scale)]
    images.extend(upscale_image(rgba_image_from_state(frame), scale) for frame in rollout_frames)
    images.extend(upscale_image(rgba_image_from_state(frame), scale) for frame in recovery_frames)
    return contact_sheet(images, columns=min(4, len(images)))


def evaluate_model(model, target, config, device, out_path=None):
    model_was_training = model.training
    model.eval()

    size = config["data"]["grid_size"]
    yy, xx = grid_cache(size, device)
    masks = build_masks(target, config)
    eval_steps = sorted(set(int(step) for step in config["eval"]["steps"]))
    preview_steps = sorted(set(int(step) for step in config["eval"]["preview_steps"]))
    max_step = max(eval_steps + preview_steps + [config["eval"]["damage_after"] + config["eval"]["recover_steps"]])

    state = make_seed_batch(1, config, device)
    state = maybe_channels_last(state, config, device)
    rollout_points = []
    rollout_frames = []
    damage_before = None

    with torch.inference_mode():
        for step in range(1, max_step + 1):
            state = model(state, steps=1)
            if step in preview_steps:
                rollout_frames.append(state.clone())
            if step in eval_steps:
                rollout_points.append({"step": step, **scalar_blob(loss_terms(state, target, masks, config))})
            if step == config["eval"]["damage_after"]:
                damage_before = state.clone()

        radius = max(2, int(round(size * config["eval"]["damage_radius_frac"])))
        recovery_start = crater_state(
            damage_before,
            radius,
            yy,
            xx,
            size // 2,
            size // 2,
        )
        recovery_frames = [recovery_start.clone()]
        recovery_state = recovery_start
        for _ in range(config["eval"]["recover_steps"]):
            recovery_state = model(recovery_state, steps=1)
        recovery_frames.append(recovery_state.clone())
        recovery_metrics = scalar_blob(loss_terms(recovery_state, target, masks, config))

    score = eval_score(rollout_points, recovery_metrics, config)
    summary = {
        "score": score,
        "recovery": recovery_metrics,
        "points": rollout_points,
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        preview_sheet(target, rollout_frames, recovery_frames, config).save(out_path)

    if model_was_training:
        model.train()
    return summary


def latest_checkpoint_path(run_dir):
    return Path(run_dir) / "checkpoints" / "latest.pt"


def best_checkpoint_path(run_dir):
    return Path(run_dir) / "checkpoints" / "best.pt"


def save_latest_checkpoint(run_dir, model, optimizer, scaler, pool, step, best_score, config):
    path = latest_checkpoint_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "best_score": best_score,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "pool": pool.detach().cpu().half(),
            "rng": checkpoint_rng_blob(),
            "config": config,
            "saved_at": now_stamp(),
        },
        path,
    )


def save_best_checkpoint(run_dir, model, step, score, summary, config):
    path = best_checkpoint_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "score": score,
            "summary": summary,
            "model": model.state_dict(),
            "config": config,
            "saved_at": now_stamp(),
        },
        path,
    )
    save_json(Path(run_dir) / "best_summary.json", {"step": step, "score": score, "summary": summary})


def try_resume(run_dir, model, optimizer, scaler, device):
    path = latest_checkpoint_path(run_dir)
    if not path.exists():
        return None

    blob = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model"])
    optimizer.load_state_dict(blob["optimizer"])
    if scaler.is_enabled() and blob.get("scaler") is not None:
        scaler.load_state_dict(blob["scaler"])
    restore_rng_blob(blob.get("rng"))
    pool = blob["pool"].float().to(device)
    return {
        "step": int(blob["step"]),
        "best_score": float(blob["best_score"]),
        "pool": pool,
    }


def train_target(config, target_path, seed):
    device = pick_device(config["runtime"].get("device", "auto"))
    setup_torch(config["runtime"], device)
    seed_everything(seed)

    run_dir = run_dir_for(config, target_path, seed)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "previews").mkdir(parents=True, exist_ok=True)

    resolved = copy.deepcopy(config)
    resolved["runtime"]["resolved_device"] = device
    resolved["target"] = str(target_path)
    resolved["seed"] = int(seed)
    save_json(run_dir / "resolved_config.json", resolved)

    target = target_tensor_from_config(target_path, config, device)
    target = maybe_channels_last(target, config, device)
    model = model_from_config(config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    amp_enabled = config["train"].get("amp", True) and device == "cuda"
    amp_dtype = dtype_from_name(config["train"].get("amp_dtype", "bfloat16"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    size = config["data"]["grid_size"]
    pool = make_seed_batch(config["train"]["pool_size"], config, device)
    pool = maybe_channels_last(pool, config, device)
    fresh_seed = make_seed_batch(1, config, device)[0]
    fresh_seed = maybe_channels_last(fresh_seed.unsqueeze(0), config, device)[0]
    yy, xx = grid_cache(size, device)
    masks = build_masks(target, config)

    start_step = 0
    best_score = math.inf
    if config["train"].get("resume", True):
        resumed = try_resume(run_dir, model, optimizer, scaler, device)
        if resumed is not None:
            start_step = resumed["step"]
            best_score = resumed["best_score"]
            pool = maybe_channels_last(resumed["pool"], config, device)
            print(f"resumed {target_path.stem} seed {seed} from step {start_step}")

    if start_step >= config["train"]["steps"]:
        print(f"{target_path.stem} seed {seed} already finished")
        return run_dir

    metrics_path = run_dir / "metrics.jsonl"
    append_jsonl(
        metrics_path,
        {
            "kind": "run_start",
            "target": target_path.stem,
            "seed": int(seed),
            "step": start_step,
            "device": device,
            "time": now_stamp(),
        },
    )

    start_time = time.time()
    for step in range(start_step + 1, config["train"]["steps"] + 1):
        progress = (step - 1) / max(config["train"]["steps"] - 1, 1)
        rollout_min, rollout_max = rollout_bounds(config, progress)
        rollout = int(torch.randint(rollout_min, rollout_max + 1, (1,), device=device).item())

        batch_ids = torch.randint(pool.shape[0], (config["train"]["batch_size"],), device=device)
        batch = pool[batch_ids].clone()
        batch[: config["train"]["fresh_seed_count"]] = fresh_seed
        batch = damage_batch(batch, config, yy, xx)
        batch = maybe_channels_last(batch, config, device)

        autocast_ctx = torch.amp.autocast(device, dtype=amp_dtype, enabled=amp_enabled) if amp_enabled else nullcontext()
        with autocast_ctx:
            batch = model(batch, steps=rollout)
            losses = loss_terms(batch, target, masks, config)
            total = losses["total"]

        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            if config["train"]["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            if config["train"]["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])
            optimizer.step()

        batch = batch.detach()
        pool[batch_ids] = batch
        worst = batch_ids[torch.argmax((batch[:, :4].clamp(0.0, 1.0) - target.expand(batch.shape[0], -1, -1, -1)).pow(2).mean(dim=(1, 2, 3)))]
        pool[worst] = fresh_seed

        if step % 25 == 0 or step == start_step + 1 or step == config["train"]["steps"]:
            elapsed = max(time.time() - start_time, 1e-6)
            speed = (step - start_step) / elapsed
            blob = {
                "kind": "train",
                "step": step,
                "rollout": rollout,
                "speed": speed,
            }
            for key, value in losses.items():
                blob[key] = float(value.detach())
            append_jsonl(metrics_path, blob)
            print(
                f"{target_path.stem} seed {seed} "
                f"step {step:5d}/{config['train']['steps']} "
                f"loss {blob['total']:.5f} rollout {rollout:3d} "
                f"speed {speed:.2f} it/s"
            )

        if step % config["train"]["eval_every"] == 0 or step == config["train"]["steps"]:
            preview_path = run_dir / "previews" / f"step_{step:05d}.png"
            summary = evaluate_model(model, target, config, device, out_path=preview_path)
            append_jsonl(
                metrics_path,
                {
                    "kind": "eval",
                    "step": step,
                    "score": summary["score"],
                    "recovery_total": summary["recovery"]["total"],
                    "tail_total": summary["points"][-1]["total"],
                    "preview": str(preview_path),
                },
            )
            print(
                f"{target_path.stem} seed {seed} eval "
                f"step {step:5d} score {summary['score']:.5f} "
                f"recovery {summary['recovery']['total']:.5f}"
            )
            if summary["score"] < best_score:
                best_score = summary["score"]
                save_best_checkpoint(run_dir, model, step, best_score, summary, resolved)

        if step % config["train"]["save_every"] == 0 or step == config["train"]["steps"]:
            save_latest_checkpoint(run_dir, model, optimizer, scaler, pool, step, best_score, resolved)

    append_jsonl(
        metrics_path,
        {
            "kind": "run_end",
            "step": config["train"]["steps"],
            "best_score": best_score,
            "time": now_stamp(),
        },
    )
    return run_dir


def resolve_config(path, overrides=None):
    config = load_config(path)
    if overrides:
        config = merge_dict(config, overrides)
    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="v2/configs/rtx5090_base.toml")
    parser.add_argument("--target", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--group-name")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    overrides = {}

    if args.steps is not None:
        overrides.setdefault("train", {})["steps"] = args.steps
    if args.device:
        overrides.setdefault("runtime", {})["device"] = args.device
    if args.group_name:
        overrides.setdefault("run", {})["group_name"] = args.group_name
    if args.batch_size is not None:
        overrides.setdefault("train", {})["batch_size"] = args.batch_size
    if args.pool_size is not None:
        overrides.setdefault("train", {})["pool_size"] = args.pool_size
    if args.no_compile:
        overrides.setdefault("train", {})["compile"] = False
    if args.no_amp:
        overrides.setdefault("train", {})["amp"] = False

    config = resolve_config(args.config, overrides)
    train_target(config, Path(args.target), args.seed)


if __name__ == "__main__":
    main()
