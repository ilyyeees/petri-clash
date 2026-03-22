import torch
from torch import nn
import torch.nn.functional as F


class NCA(nn.Module):
    def __init__(self, channels=16, hidden_size=128, fire_rate=0.5):
        super().__init__()
        self.channels = channels
        self.hidden_size = hidden_size
        self.fire_rate = fire_rate

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ) / 8.0
        sobel_y = sobel_x.t()
        identity = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )

        filters = torch.stack([identity, sobel_x, sobel_y], dim=0)[:, None]
        filters = filters.repeat(channels, 1, 1, 1)
        self.register_buffer("perception_filters", filters)

        self.fc0 = nn.Conv2d(channels * 3, hidden_size, 1)
        self.fc1 = nn.Conv2d(hidden_size, channels, 1)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

    def perceive(self, x):
        return F.conv2d(x, self.perception_filters, padding=1, groups=self.channels)

    def alive_mask(self, x):
        alpha = x[:, 3:4]
        return F.max_pool2d(alpha, 3, stride=1, padding=1) > 0.1

    def step(self, x, fire_rate=None):
        if fire_rate is None:
            fire_rate = self.fire_rate

        pre_life = self.alive_mask(x)
        y = self.perceive(x)
        dx = self.fc1(F.relu(self.fc0(y)))

        if fire_rate < 1.0:
            # this keeps updates patchy so the thing has to work asynchronously
            mask = (torch.rand_like(x[:, :1]) <= fire_rate).float()
            dx = dx * mask

        x = x + dx
        post_life = self.alive_mask(x)
        x = x * (pre_life | post_life).float()
        return x

    def forward(self, x, steps=1, fire_rate=None):
        for _ in range(steps):
            x = self.step(x, fire_rate=fire_rate)
        return x


def make_seed(batch_size, channels=16, height=48, width=None, xs=None, ys=None, device=None):
    if width is None:
        width = height

    state = torch.zeros(batch_size, channels, height, width, device=device)
    if xs is None:
        xs = torch.full((batch_size,), width // 2, dtype=torch.long, device=device)
    else:
        xs = torch.as_tensor(xs, dtype=torch.long, device=device)
    if ys is None:
        ys = torch.full((batch_size,), height // 2, dtype=torch.long, device=device)
    else:
        ys = torch.as_tensor(ys, dtype=torch.long, device=device)

    batch_ids = torch.arange(batch_size, device=device)

    # one alpha pulse and one hidden spark is enough to get things moving
    state[batch_ids, 3, ys, xs] = 1.0
    state[batch_ids, 4, ys, xs] = 1.0
    return state


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
