from typing import Any

import torch
import torchvision
import tqdm
from loguru import logger

logger.remove()
logger.add(lambda msg: tqdm.tqdm.write(msg, end=""), colorize=True)


# CNNモデル
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(64 * 7 * 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


def main() -> None:
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
            batch_size=128,
            shuffle=True,
            num_workers=12,
        )
    )

    test_loader: torch.utils.data.DataLoader[tuple[Any, Any]] = (
        torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                root="./data", train=False, download=True, transform=transform
            ),
            batch_size=128,
            shuffle=True,
            num_workers=12,
        )
    )

    model = Model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training
    for epoch in tqdm.tqdm(range(10)):
        total_loss = 0

        for images, labels in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

    # Evaluation
    correct = 0
    incorrect = 0

    with torch.no_grad():
        model.eval()
        for images, labels in tqdm.tqdm(test_loader):
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            incorrect += (predicted != labels).sum().item()

    logger.info(f"Accuracy: {correct / (correct + incorrect) * 100:.2f}%")


if __name__ == "__main__":
    main()
