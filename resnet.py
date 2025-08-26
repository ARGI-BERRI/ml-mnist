from argparse import ArgumentParser
from typing import Any

import torch
import torchvision
import tqdm
from loguru import logger

logger.remove()
logger.add(lambda msg: tqdm.tqdm.write(msg, end=""), colorize=True)


class Model(torch.nn.Module):
    """A ResNet18 model for image classification."""

    def __init__(self) -> None:
        super().__init__()

        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


def get_torch_device() -> torch.device:
    """Get the current PyTorch device.
    - `cuda` if the GPU is NVIDIA or AMD (with ROCm support)
    - `xpu` if the GPU is Intel
    - `cpu` otherwise

    Returns:
        torch.device: The current PyTorch device.
    """
    # NVIDIA or AMD GPU
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Intel GPU
    if torch.xpu.is_available():
        return torch.device("xpu")

    return torch.device("cpu")


def main(batch_size: int, num_of_workers: int, epochs: int) -> None:
    device = get_torch_device()
    logger.debug(f"Using device: {device}")
    logger.debug(f"{batch_size=}, {num_of_workers=}, {epochs=}")

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_loader: torch.utils.data.DataLoader[tuple[Any, Any]] = (
        torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                root="./data", train=True, download=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_of_workers,
            pin_memory=True,
            prefetch_factor=4,
        )
    )

    test_loader: torch.utils.data.DataLoader[tuple[Any, Any]] = (
        torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                root="./data", train=False, download=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_of_workers,
            pin_memory=True,
            prefetch_factor=4,
        )
    )

    logger.success("Data loaders have been initialized")

    model = Model().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    logger.success("Model has been initialized")

    # Training
    logger.info("Starting training...")
    model.train()
    for epoch in tqdm.tqdm(range(epochs), desc="Training"):
        total_loss = 0

        scaler = torch.amp.grad_scaler.GradScaler(device=device.type)
        for images, labels in tqdm.tqdm(train_loader, desc=f"Training #{epoch + 1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast_mode.autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        logger.info(f"Epoch {epoch + 1} done, Loss: {total_loss / len(train_loader)}")

    # Evaluation
    logger.info("Starting evaluation...")
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    correct = 0
    incorrect = 0

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm.tqdm(test_loader, desc="Evaluation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            incorrect += (predicted != labels).sum().item()

            for i in range(len(labels)):
                if labels[i] == 1:
                    if predicted[i] == 1:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if predicted[i] == 1:
                        fp += 1
                    else:
                        tn += 1

    logger.info(f"Result: {tp=}, {tn=}, {fp=}, {fn=}")

    accuracy = correct / (correct + incorrect) * 100
    logger.info(f"Accuracy: {accuracy:.4f}%")

    f1_score = (2 * tp) / (2 * tp + fp + fn)
    logger.info(f"F1 Score: {f1_score:.4f}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-w", "--num_of_workers", type=int, default=8)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    args = parser.parse_args()
    main(
        batch_size=args.batch_size,
        num_of_workers=args.num_of_workers,
        epochs=args.epochs,
    )
