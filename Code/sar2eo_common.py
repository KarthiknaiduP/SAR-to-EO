"""Shared, reproducible components for the SAR-to-EO Kaggle experiments."""

from __future__ import annotations

import csv
import json
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


DEFAULT_DATA_ROOT = "/kaggle/input/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain/v_2"
DEFAULT_OUT = "/kaggle/working/sar2eo_outputs"
DEFAULT_MANIFEST = f"{DEFAULT_OUT}/manifest.csv"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_manifest(path: str | Path, split: str | None = None, limit: int | None = None) -> list[dict[str, str]]:
    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle))
    if split:
        rows = [row for row in rows if row["split"] == split]
    return rows[:limit] if limit else rows


def load_sar(path: str, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("L").resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)[None] / 127.5 - 1.0
    return torch.from_numpy(array)


def load_rgb(path: str, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0
    return torch.from_numpy(array)


def augment_pair(sar: torch.Tensor, eo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        sar, eo = sar.flip(2), eo.flip(2)
    if random.random() < 0.5:
        sar, eo = sar.flip(1), eo.flip(1)
    turns = random.randint(0, 3)
    if turns:
        sar, eo = torch.rot90(sar, turns, (1, 2)), torch.rot90(eo, turns, (1, 2))
    return sar.contiguous(), eo.contiguous()


class SarOpticalDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        split: str,
        image_size: int = 256,
        augment: bool = False,
        limit: int | None = None,
        unpaired: bool = False,
        seed: int = 42,
    ):
        self.rows = load_manifest(manifest, split, limit)
        if not self.rows:
            raise ValueError(f"No rows for split={split} in {manifest}")
        self.image_size = image_size
        self.augment = augment
        self.unpaired = unpaired
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        sar_row = self.rows[index]
        # A deterministic offset makes CycleGAN unpaired while remaining reproducible.
        eo_index = (index * 7919 + self.seed) % len(self.rows) if self.unpaired else index
        eo_row = self.rows[eo_index]
        sar = load_sar(sar_row["sar_path"], self.image_size)
        eo = load_rgb(eo_row["eo_path"], self.image_size)
        if self.augment:
            sar, eo = augment_pair(sar, eo)
        return {
            "sar": sar,
            "eo": eo,
            "sar_path": sar_row["sar_path"],
            "eo_path": eo_row["eo_path"],
            "scene_id": sar_row["scene_id"],
            "terrain": sar_row.get("terrain", "unknown"),
        }


def make_loaders(config: "TrainConfig", unpaired: bool = False) -> tuple[DataLoader, DataLoader]:
    train = SarOpticalDataset(
        config.manifest, "train", config.image_size, True, config.train_limit, unpaired, config.seed
    )
    val = SarOpticalDataset(config.manifest, "val", config.image_size, False, config.val_limit)
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    return train_loader, val_loader


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    """Pix2Pix U-Net generator for 256 x 256 inputs."""

    def __init__(self, in_channels: int = 1, out_channels: int = 3, base: int = 64):
        super().__init__()
        b = base
        self.downs = nn.ModuleList(
            [
                Down(in_channels, b, False),
                Down(b, b * 2),
                Down(b * 2, b * 4),
                Down(b * 4, b * 8),
                Down(b * 8, b * 8),
                Down(b * 8, b * 8),
                Down(b * 8, b * 8),
            ]
        )
        self.bottleneck = nn.Sequential(nn.Conv2d(b * 8, b * 8, 4, 2, 1), nn.ReLU(True))
        self.ups = nn.ModuleList(
            [
                Up(b * 8, b * 8, 0.5),
                Up(b * 16, b * 8, 0.5),
                Up(b * 16, b * 8, 0.5),
                Up(b * 16, b * 8),
                Up(b * 16, b * 4),
                Up(b * 8, b * 2),
                Up(b * 4, b),
            ]
        )
        self.final = nn.Sequential(nn.ConvTranspose2d(b * 2, out_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = self.bottleneck(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x)
            x = torch.cat((x, skip), dim=1)
        return self.final(x)


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.channel = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.ReLU(True), nn.Conv2d(hidden, channels, 1)
        )
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(
            self.channel(F.adaptive_avg_pool2d(x, 1)) + self.channel(F.adaptive_max_pool2d(x, 1))
        )
        x = x * gate
        spatial = torch.cat((x.mean(1, keepdim=True), x.amax(1, keepdim=True)), dim=1)
        return x * torch.sigmoid(self.spatial(spatial))


class ResidualAttentionBlock(nn.Module):
    def __init__(self, channels: int, attention: bool = True):
        super().__init__()
        self.body = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )
        self.attention = CBAM(channels) if attention else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x + self.body(x))


class AttentionResUNetGenerator(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 3, base: int = 64, attention: bool = True):
        super().__init__()
        b = base
        self.d1, self.d2 = Down(in_channels, b, False), Down(b, b * 2)
        self.d3, self.d4 = Down(b * 2, b * 4), Down(b * 4, b * 8)
        self.d5, self.d6 = Down(b * 8, b * 8), Down(b * 8, b * 8)
        self.bottom = nn.Sequential(
            Down(b * 8, b * 8),
            ResidualAttentionBlock(b * 8, attention),
            ResidualAttentionBlock(b * 8, attention),
        )
        self.u1, self.u2 = Up(b * 8, b * 8, 0.5), Up(b * 16, b * 8, 0.5)
        self.u3, self.u4 = Up(b * 16, b * 8), Up(b * 16, b * 4)
        self.u5, self.u6 = Up(b * 8, b * 2), Up(b * 4, b)
        self.final = nn.Sequential(nn.ConvTranspose2d(b * 2, out_channels, 4, 2, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        x = self.bottom(d6)
        x = self.u1(x)
        x = self.u2(torch.cat((x, d6), 1))
        x = self.u3(torch.cat((x, d5), 1))
        x = self.u4(torch.cat((x, d4), 1))
        x = self.u5(torch.cat((x, d3), 1))
        x = self.u6(torch.cat((x, d2), 1))
        return self.final(torch.cat((x, d1), 1))


class PatchDiscriminator(nn.Module):
    def __init__(self, condition_channels: int = 1, image_channels: int = 3, base: int = 64, spectral: bool = False):
        super().__init__()

        def conv(*args, **kwargs):
            layer = nn.Conv2d(*args, **kwargs)
            return spectral_norm(layer) if spectral else layer

        b = base
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(conv(condition_channels + image_channels, b, 4, 2, 1), nn.LeakyReLU(0.2, True)),
                nn.Sequential(conv(b, b * 2, 4, 2, 1, bias=False), nn.InstanceNorm2d(b * 2), nn.LeakyReLU(0.2, True)),
                nn.Sequential(conv(b * 2, b * 4, 4, 2, 1, bias=False), nn.InstanceNorm2d(b * 4), nn.LeakyReLU(0.2, True)),
                nn.Sequential(conv(b * 4, b * 8, 4, 1, 1, bias=False), nn.InstanceNorm2d(b * 8), nn.LeakyReLU(0.2, True)),
            ]
        )
        self.final = conv(b * 8, 1, 4, 1, 1)

    def forward(self, condition: torch.Tensor, image: torch.Tensor, return_features: bool = False):
        x = torch.cat((condition, image), dim=1)
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        output = self.final(x)
        return (output, features) if return_features else output


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    """CycleGAN ResNet generator, supporting unequal source/target channels."""

    def __init__(self, in_channels: int, out_channels: int, base: int = 64, blocks: int = 6):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base, 7, bias=False),
            nn.InstanceNorm2d(base),
            nn.ReLU(True),
        ]
        channels = base
        for _ in range(2):
            layers += [
                nn.Conv2d(channels, channels * 2, 3, 2, 1, bias=False),
                nn.InstanceNorm2d(channels * 2),
                nn.ReLU(True),
            ]
            channels *= 2
        layers += [ResidualBlock(channels) for _ in range(blocks)]
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(channels, channels // 2, 3, 2, 1, output_padding=1, bias=False),
                nn.InstanceNorm2d(channels // 2),
                nn.ReLU(True),
            ]
            channels //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(channels, out_channels, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DomainDiscriminator(nn.Module):
    def __init__(self, channels: int, base: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, base, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(base * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base * 4, base * 8, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(base * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(base * 8, 1, 4, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceptualLoss(nn.Module):
    """VGG feature loss with a no-download multiscale fallback."""

    def __init__(self, device: torch.device):
        super().__init__()
        self.mode = "multiscale_fallback"
        self.features: nn.Module | None = None
        try:
            from torchvision.models import VGG16_Weights, vgg16

            self.features = vgg16(weights=VGG16_Weights.DEFAULT).features[:23].eval().to(device)
            self.features.requires_grad_(False)
            self.mode = "vgg16_imagenet"
        except Exception as error:
            print(f"VGG weights unavailable ({error!r}); using multiscale feature loss.")
        self.layers = {3, 8, 15, 22}
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.features is None:
            loss = prediction.new_tensor(0.0)
            pred, real = prediction, target
            for _ in range(3):
                loss = loss + F.l1_loss(pred, real)
                pred, real = F.avg_pool2d(pred, 2), F.avg_pool2d(real, 2)
            return loss / 3.0
        pred = ((prediction + 1) / 2 - self.mean) / self.std
        real = ((target + 1) / 2 - self.mean) / self.std
        loss = prediction.new_tensor(0.0)
        for index, layer in enumerate(self.features):
            pred, real = layer(pred), layer(real)
            if index in self.layers:
                loss = loss + F.l1_loss(pred, real)
        return loss


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def feature_matching(fake: list[torch.Tensor], real: list[torch.Tensor]) -> torch.Tensor:
    return sum(F.l1_loss(a, b.detach()) for a, b in zip(fake, real)) / len(fake)


def gradient_consistency_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    real_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    real_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return 0.5 * (F.l1_loss(pred_dx, real_dx) + F.l1_loss(pred_dy, real_dy))


def update_ema(source: nn.Module, target: nn.Module, decay: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(decay).add_(source_param.data, alpha=1.0 - decay)
        for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
            target_buffer.data.copy_(source_buffer.data)


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = (tensor.detach().cpu().clamp(-1, 1) + 1) / 2
    return tensor[0].numpy() if tensor.shape[0] == 1 else tensor.permute(1, 2, 0).numpy()


def save_samples(generator: nn.Module, loader: DataLoader, device: torch.device, path: Path, count: int = 5) -> None:
    generator.eval()
    batch = next(iter(loader))
    sar, real = batch["sar"].to(device), batch["eo"].to(device)
    with torch.inference_mode():
        fake = generator(sar)
    count = min(count, len(sar))
    figure, axes = plt.subplots(count, 3, figsize=(9, count * 3), squeeze=False)
    for index in range(count):
        axes[index, 0].imshow(tensor_to_image(sar[index]), cmap="gray")
        axes[index, 1].imshow(tensor_to_image(fake[index]))
        axes[index, 2].imshow(tensor_to_image(real[index]))
        for axis, title in zip(axes[index], ("SAR", "Generated EO", "Target EO")):
            axis.set_title(title)
            axis.axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_history(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(rows: list[dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    for key in rows[0]:
        if key not in {"epoch", "seconds"}:
            axis.plot([row["epoch"] for row in rows], [row[key] for row in rows], label=key)
    axis.set(xlabel="Epoch", ylabel="Loss")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


@dataclass
class TrainConfig:
    model_name: str
    run_name: str
    manifest: str = DEFAULT_MANIFEST
    out_dir: str = DEFAULT_OUT
    epochs: int = 30
    batch_size: int = 4
    num_workers: int = 2
    image_size: int = 256
    base_channels: int = 64
    train_limit: int | None = 8000
    val_limit: int | None = 1000
    learning_rate: float = 2e-4
    lambda_l1: float = 100.0
    lambda_perceptual: float = 0.0
    lambda_feature_matching: float = 0.0
    lambda_gradient: float = 0.0
    lambda_cycle: float = 10.0
    gan_mode: str = "bce"
    attention: bool = True
    amp: bool = True
    seed: int = 42
    warm_start_checkpoint: str | None = None
    ema_decay: float = 0.0


def build_generator(config: dict | TrainConfig) -> nn.Module:
    cfg = asdict(config) if isinstance(config, TrainConfig) else config
    name = cfg["model_name"]
    base = int(cfg.get("base_channels", 64))
    if name in {"unet_l1", "pix2pix", "pix2pix_perceptual"}:
        return UNetGenerator(base=base)
    if name in {"attention_resunet", "final_fusion_gan"}:
        return AttentionResUNetGenerator(base=base, attention=bool(cfg.get("attention", True)))
    if name == "cyclegan":
        return ResNetGenerator(1, 3, base=base, blocks=6)
    raise ValueError(f"Unknown model_name={name}")


def save_checkpoint(
    path: Path,
    epoch: int,
    generator: nn.Module,
    config: TrainConfig,
    validation_l1: float,
    discriminator: nn.Module | None = None,
    extra: dict | None = None,
) -> None:
    payload = {
        "format_version": 2,
        "epoch": epoch,
        "model_name": config.model_name,
        "generator": generator.state_dict(),
        "config": asdict(config),
        "validation_l1": validation_l1,
    }
    if discriminator is not None:
        payload["discriminator"] = discriminator.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_generator(checkpoint: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    payload = torch.load(checkpoint, map_location=device)
    config = payload["config"]
    config.setdefault("model_name", payload.get("model_name"))
    generator = build_generator(config)
    generator.load_state_dict(payload["generator"])
    return generator.to(device).eval(), payload


def validate_l1(generator: nn.Module, loader: DataLoader, device: torch.device) -> float:
    generator.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            sar, target = batch["sar"].to(device), batch["eo"].to(device)
            total += F.l1_loss(generator(sar), target, reduction="sum").item()
            count += target.numel()
    return total / count


def maybe_warm_start_generator(generator: nn.Module, checkpoint: str | None, device: torch.device) -> None:
    if not checkpoint:
        return
    path = Path(checkpoint)
    if not path.exists():
        print(f"Warm start skipped; checkpoint not found: {path}")
        return
    payload = torch.load(path, map_location=device)
    try:
        generator.load_state_dict(payload["generator"], strict=True)
        source = payload.get("config", {}).get("run_name", str(path))
        print(f"Warm started generator from {source}: {path}")
    except RuntimeError as error:
        print(f"Warm start skipped; checkpoint architecture is incompatible: {error}")


def train_paired(config: TrainConfig) -> Path:
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(config.out_dir) / config.run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = make_loaders(config)
    generator = build_generator(config).to(device)
    generator.apply(initialize_weights)
    maybe_warm_start_generator(generator, config.warm_start_checkpoint, device)
    ema_generator = None
    if config.ema_decay > 0:
        ema_generator = build_generator(config).to(device)
        ema_generator.load_state_dict(generator.state_dict())
        ema_generator.eval()
    use_gan = config.model_name != "unet_l1"
    discriminator = None
    if use_gan:
        discriminator = PatchDiscriminator(
            base=config.base_channels,
            spectral=config.gan_mode == "hinge",
        ).to(device)
        discriminator.apply(initialize_weights)
    perceptual = PerceptualLoss(device).to(device) if config.lambda_perceptual else None
    optimizer_g = torch.optim.Adam(generator.parameters(), config.learning_rate, betas=(0.5, 0.999))
    optimizer_d = (
        torch.optim.Adam(discriminator.parameters(), config.learning_rate, betas=(0.5, 0.999))
        if discriminator
        else None
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and config.amp)
    history, best = [], float("inf")
    (run_dir / "config_used.json").write_text(json.dumps(asdict(config), indent=2))
    print(f"{config.model_name}: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, device={device}")

    for epoch in range(1, config.epochs + 1):
        started = time.time()
        generator.train()
        if discriminator:
            discriminator.train()
        totals: defaultdict[str, float] = defaultdict(float)
        seen = 0
        progress = tqdm(train_loader, desc=f"{config.run_name} epoch {epoch}/{config.epochs}")
        for batch in progress:
            sar = batch["sar"].to(device, non_blocking=True)
            target = batch["eo"].to(device, non_blocking=True)
            batch_size = len(sar)
            if discriminator and optimizer_d:
                optimizer_d.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    with torch.no_grad():
                        detached = generator(sar)
                    real_logits = discriminator(sar, target)
                    fake_logits = discriminator(sar, detached)
                    if config.gan_mode == "hinge":
                        d_loss = F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
                    else:
                        d_loss = 0.5 * (
                            F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
                            + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
                        )
                scaler.scale(d_loss).backward()
                scaler.step(optimizer_d)
            else:
                d_loss = sar.new_tensor(0.0)

            optimizer_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                prediction = generator(sar)
                l1 = F.l1_loss(prediction, target)
                adversarial = sar.new_tensor(0.0)
                fm_loss = sar.new_tensor(0.0)
                if discriminator:
                    fake_logits, fake_features = discriminator(sar, prediction, True)
                    if config.gan_mode == "hinge":
                        adversarial = -fake_logits.mean()
                    else:
                        adversarial = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
                    if config.lambda_feature_matching:
                        with torch.no_grad():
                            _, real_features = discriminator(sar, target, True)
                        fm_loss = feature_matching(fake_features, real_features)
                perceptual_loss = perceptual(prediction, target) if perceptual else sar.new_tensor(0.0)
                gradient_loss = (
                    gradient_consistency_loss(prediction, target)
                    if config.lambda_gradient
                    else sar.new_tensor(0.0)
                )
                g_loss = (
                    adversarial
                    + config.lambda_l1 * l1
                    + config.lambda_perceptual * perceptual_loss
                    + config.lambda_feature_matching * fm_loss
                    + config.lambda_gradient * gradient_loss
                )
            scaler.scale(g_loss).backward()
            scaler.step(optimizer_g)
            scaler.update()
            if ema_generator is not None:
                update_ema(generator, ema_generator, config.ema_decay)
            seen += batch_size
            for key, value in {
                "g_loss": g_loss,
                "d_loss": d_loss,
                "l1": l1,
                "adversarial": adversarial,
                "perceptual": perceptual_loss,
                "feature_matching": fm_loss,
                "gradient": gradient_loss,
            }.items():
                totals[key] += float(value.detach()) * batch_size
            progress.set_postfix(g=f"{g_loss.item():.3f}", d=f"{d_loss.item():.3f}", l1=f"{l1.item():.3f}")

        validation_model = ema_generator if ema_generator is not None else generator
        val_l1 = validate_l1(validation_model, val_loader, device)
        row = {"epoch": epoch, **{key: value / seen for key, value in totals.items()}, "val_l1": val_l1, "seconds": time.time() - started}
        history.append(row)
        write_history(history, run_dir / "losses.csv")
        plot_history(history, run_dir / "loss_curves.png")
        save_samples(validation_model, val_loader, device, run_dir / f"samples_epoch_{epoch:03d}.png")
        save_checkpoint(checkpoint_dir / "latest.pt", epoch, validation_model, config, val_l1, discriminator)
        if val_l1 < best:
            best = val_l1
            save_checkpoint(checkpoint_dir / "best.pt", epoch, validation_model, config, val_l1, discriminator)
        print(json.dumps(row, indent=2))
    return checkpoint_dir / "best.pt"


def train_cyclegan(config: TrainConfig) -> Path:
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(config.out_dir) / config.run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = make_loaders(config, unpaired=True)
    g_sar_to_eo = ResNetGenerator(1, 3, config.base_channels, 6).to(device)
    g_eo_to_sar = ResNetGenerator(3, 1, config.base_channels, 6).to(device)
    d_eo, d_sar = DomainDiscriminator(3, config.base_channels).to(device), DomainDiscriminator(1, config.base_channels).to(device)
    for model in (g_sar_to_eo, g_eo_to_sar, d_eo, d_sar):
        model.apply(initialize_weights)
    opt_g = torch.optim.Adam(
        list(g_sar_to_eo.parameters()) + list(g_eo_to_sar.parameters()), config.learning_rate, betas=(0.5, 0.999)
    )
    opt_d = torch.optim.Adam(list(d_eo.parameters()) + list(d_sar.parameters()), config.learning_rate, betas=(0.5, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and config.amp)
    history, best = [], float("inf")
    (run_dir / "config_used.json").write_text(json.dumps(asdict(config), indent=2))

    for epoch in range(1, config.epochs + 1):
        started = time.time()
        totals: defaultdict[str, float] = defaultdict(float)
        seen = 0
        progress = tqdm(train_loader, desc=f"{config.run_name} epoch {epoch}/{config.epochs}")
        for batch in progress:
            real_sar, real_eo = batch["sar"].to(device), batch["eo"].to(device)
            batch_size = len(real_sar)
            opt_g.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                fake_eo, fake_sar = g_sar_to_eo(real_sar), g_eo_to_sar(real_eo)
                cycle_sar, cycle_eo = g_eo_to_sar(fake_eo), g_sar_to_eo(fake_sar)
                gan_g = 0.5 * ((d_eo(fake_eo) - 1).square().mean() + (d_sar(fake_sar) - 1).square().mean())
                cycle = F.l1_loss(cycle_sar, real_sar) + F.l1_loss(cycle_eo, real_eo)
                g_loss = gan_g + config.lambda_cycle * cycle
            scaler.scale(g_loss).backward()
            scaler.step(opt_g)

            opt_d.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                d_eo_loss = 0.5 * ((d_eo(real_eo) - 1).square().mean() + d_eo(fake_eo.detach()).square().mean())
                d_sar_loss = 0.5 * ((d_sar(real_sar) - 1).square().mean() + d_sar(fake_sar.detach()).square().mean())
                d_loss = 0.5 * (d_eo_loss + d_sar_loss)
            scaler.scale(d_loss).backward()
            scaler.step(opt_d)
            scaler.update()
            seen += batch_size
            for key, value in {"g_loss": g_loss, "d_loss": d_loss, "cycle": cycle, "adversarial": gan_g}.items():
                totals[key] += float(value.detach()) * batch_size
            progress.set_postfix(g=f"{g_loss.item():.3f}", d=f"{d_loss.item():.3f}", cycle=f"{cycle.item():.3f}")

        val_l1 = validate_l1(g_sar_to_eo, val_loader, device)
        row = {"epoch": epoch, **{key: value / seen for key, value in totals.items()}, "val_l1": val_l1, "seconds": time.time() - started}
        history.append(row)
        write_history(history, run_dir / "losses.csv")
        plot_history(history, run_dir / "loss_curves.png")
        save_samples(g_sar_to_eo, val_loader, device, run_dir / f"samples_epoch_{epoch:03d}.png")
        extra = {"reverse_generator": g_eo_to_sar.state_dict(), "eo_discriminator": d_eo.state_dict(), "sar_discriminator": d_sar.state_dict()}
        save_checkpoint(checkpoint_dir / "latest.pt", epoch, g_sar_to_eo, config, val_l1, extra=extra)
        if val_l1 < best:
            best = val_l1
            save_checkpoint(checkpoint_dir / "best.pt", epoch, g_sar_to_eo, config, val_l1, extra=extra)
        print(json.dumps(row, indent=2))
    return checkpoint_dir / "best.pt"
