import torch
import torch.nn as nn

try:
    from torchinfo import summary
except ImportError:
    summary = None


class SpokenDigitCNN(nn.Module):
    """Baseline Model: Standard 2D CNN with a dense hidden classifier."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.35):
        super().__init__()

        self.features = nn.Sequential(
            # Input: [B, 1, 40, 63]
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # [B, 16, 20, 31]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # [B, 32, 10, 15]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Preserve a small spatial representation
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SpokenDigitGAP(nn.Module):
    """Experimental Model 1: Replaces dense hidden layer with Global Average Pooling (GAP).
    Drastically cuts down parameter count and mitigates classifier overfitting."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.2):
        super().__init__()

        self.features = nn.Sequential(
            # Input: [B, 1, 40, 63]
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # [B, 16, 20, 31]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # [B, 32, 10, 15]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Collapse spatial dimension completely to (1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            # Directly mapping feature maps to class logits (no heavy intermediate 128-dim linear layer)
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class DepthwiseSeparableConv2d(nn.Module):
    """Helper block to implement computationally ultra-lean depthwise separable convolutions."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1, bias: bool = True):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SpokenDigitMobile(nn.Module):
    """Experimental Model 2: Uses Depthwise Separable Convolutions + GAP to target minimal MACs/FLOPs."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.2):
        super().__init__()

        self.features = nn.Sequential(
            # Input: [B, 1, 40, 63] - Keep first layer standard for rich low-level spectrogram representation
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Depthwise Separable Block 1
            DepthwiseSeparableConv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Depthwise Separable Block 2
            DepthwiseSeparableConv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


MODELS = {
    "SpokenDigitCNN": SpokenDigitCNN,
    "SpokenDigitGAP": SpokenDigitGAP,
    "SpokenDigitMobile": SpokenDigitMobile,
    "Baseline CNN": SpokenDigitCNN,
    "CNN + GAP": SpokenDigitGAP,
    "MobileNet-style (Depthwise + GAP)": SpokenDigitMobile,
}


def get_model(name: str = "SpokenDigitCNN", **kwargs):
    """Factory function to instantiate models by name."""
    if name not in MODELS:
        raise ValueError(
            f"Unknown model architecture '{name}'. Available options: {list(MODELS.keys())}"
        )
    return MODELS[name](**kwargs)


def count_parameters(model):
    """Return total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimated_parameter_size_mb(model):
    """Approximate FP32 parameter size in megabytes."""
    return count_parameters(model) * 4 / (1024 ** 2)


# --- Profiling Runner for Architectural Comparison ---
if __name__ == "__main__":
    # Test batch shape matching audio frontend configuration: [Batch, Channel, Mels, Time_Steps]
    input_size = (1, 1, 40, 63)

    models_to_profile = {
        "Baseline CNN": SpokenDigitCNN(),
        "CNN + GAP Optimization": SpokenDigitGAP(),
        "Depthwise Separable + GAP": SpokenDigitMobile(),
    }

    print("=" * 68)
    print("ARCHITECTURAL EFFICIENCY COMPARISON METRICS")
    print("=" * 68)

    for name, model in models_to_profile.items():
        total_p = count_parameters(model)
        size_kb = (total_p * 4) / 1024
        print(f"\nModel Variant: {name}")
        print(f" -> Total Parameters   : {total_p:,}")
        print(f" -> Parameter Memory   : {size_kb:.2f} KB ({size_kb / 1024:.4f} MB)")

        if summary is not None:
            try:
                stats = summary(model, input_size=input_size, verbose=0)
                print(f" -> Mult-Adds (MACs)   : {stats.total_mult_adds:,}")
            except Exception as e:
                print(f" -> MACs calculation   : (skipped: {e})")
        else:
            print(" -> Mult-Adds (MACs)   : (install torchinfo to view)")

    print("\n" + "=" * 68)