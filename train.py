import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nca import NCA, make_seed, pick_device


def load_target(path, target_size=40, grid_size=48, device="cpu"):
    image = Image.open(path).convert("RGBA")
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (grid_size, grid_size), (0, 0, 0, 0))
    offset = ((grid_size - target_size) // 2, (grid_size - target_size) // 2)
    canvas.paste(image, offset, image)

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def save_checkpoint(model, out_path, args, target_path, train_steps):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "state_dict": model.state_dict(),
        "target_name": target_path.stem,
        "target_path": str(target_path),
        "train_steps": train_steps,
        "grid_size": args.grid_size,
        "target_size": args.target_size,
        "channels": model.channels,
        "hidden_size": model.hidden_size,
        "fire_rate": model.fire_rate,
    }
    torch.save(blob, out_path)


def train_target(
    target_path,
    out_path=None,
    steps=3000,
    pool_size=1024,
    batch_size=8,
    grid_size=48,
    target_size=40,
    min_rollout=64,
    max_rollout=96,
    lr=2e-3,
    save_every=250,
    seed=0,
    device=None,
):
    if device is None:
        device = pick_device()

    target_path = Path(target_path)
    if not target_path.exists():
        raise SystemExit(f"missing target: {target_path}")

    if out_path is None:
        out_path = Path("weights") / f"{target_path.stem}.pt"
    else:
        out_path = Path(out_path)

    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    args = argparse.Namespace(
        grid_size=grid_size,
        target_size=target_size,
    )

    target = load_target(
        target_path,
        target_size=target_size,
        grid_size=grid_size,
        device=device,
    )

    model = NCA().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pool = make_seed(
        pool_size,
        height=grid_size,
        width=grid_size,
        device=device,
    )
    fresh_seed = make_seed(
        1,
        height=grid_size,
        width=grid_size,
        device=device,
    )[0]

    save_every = max(1, save_every)

    print(f"training {target_path.name} on {device}")
    start = time.time()
    ema = None

    for step in range(1, steps + 1):
        batch_ids = torch.randint(pool_size, (batch_size,), device=device)
        batch = pool[batch_ids].clone()

        # one fresh seed in the batch keeps the organism honest
        batch[0] = fresh_seed

        rollout = random.randint(min_rollout, max_rollout)
        batch = model(batch, steps=rollout)
        losses = (batch[:, :4] - target).pow(2).mean(dim=(1, 2, 3))
        loss = losses.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch = batch.detach()
        pool[batch_ids] = batch
        pool[batch_ids[losses.argmax()]] = fresh_seed

        value = float(loss.detach())
        ema = value if ema is None else ema * 0.98 + value * 0.02

        if step % 50 == 0 or step == 1 or step == steps:
            elapsed = max(time.time() - start, 1e-6)
            speed = step / elapsed
            print(
                f"step {step:4d}/{steps} loss {value:.5f} ema {ema:.5f} "
                f"rollout {rollout:2d} speed {speed:.2f} it/s"
            )

        if step % save_every == 0 or step == steps:
            save_checkpoint(model, out_path, args, target_path, step)

    print(f"saved {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--out")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--pool-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grid-size", type=int, default=48)
    parser.add_argument("--target-size", type=int, default=40)
    parser.add_argument("--min-rollout", type=int, default=64)
    parser.add_argument("--max-rollout", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=pick_device())
    args = parser.parse_args()

    train_target(
        args.target,
        out_path=args.out,
        steps=args.steps,
        pool_size=args.pool_size,
        batch_size=args.batch_size,
        grid_size=args.grid_size,
        target_size=args.target_size,
        min_rollout=args.min_rollout,
        max_rollout=args.max_rollout,
        lr=args.lr,
        save_every=args.save_every,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
