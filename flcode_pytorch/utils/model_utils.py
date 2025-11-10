from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2

from common.loggers import info


class Net(nn.Module):
    def __init__(self, num_classes: int = 62, input_channels: int = 1, image_size: int = 96, width_mult: float = 1.0):
        super(Net, self).__init__()
        self.target_hw = image_size
        self.input_channels = input_channels
        self.channel_proj = None
        if input_channels not in (1, 3):
            self.channel_proj = nn.Conv2d(input_channels, 3, kernel_size=1, bias=False)

        backbone = mobilenet_v2(weights=None, width_mult=width_mult)
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, num_classes)
        self.backbone = backbone

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.target_hw:
            x = F.interpolate(x, size=(self.target_hw, self.target_hw), mode="bilinear", align_corners=False)

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            if self.channel_proj is None:
                self.channel_proj = nn.Conv2d(x.shape[1], 3, kernel_size=1, bias=False).to(x.device)
            x = self.channel_proj(x)

        return x

    def forward(self, x):
        x = self._prepare_input(x)
        return self.backbone(x)


def train(
        model: nn.Module,
        trainloader,
        device,
        input_features: list[str],
        target_features: list[str],
        epochs: int = 1,
        learning_rate: float = 0.01,
        scheduler=None,
        log_interval: int = 100,
        **kwargs
):
    """Train the model for a number of epochs."""
    optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    model.to(device)
    model.train()
    running_loss = 0.0
    for epoch in range(epochs):
        for i, batch in enumerate(trainloader):
            inputs = batch[input_features[0]].to(device)
            targets = batch[target_features[0]].to(device)
            optim.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optim.step()
            running_loss += loss.item()
            if i % log_interval == 0:
                info(f"Epoch {epoch} - Step {i}")
        if scheduler is not None:
            scheduler.step()
    train_loss = running_loss / (epochs * len(trainloader.dataset))  # Average loss per sample
    return train_loss


def test(
        model: nn.Module,
        testloader,
        device,
        input_features: list[str],
        target_features: list[str],
        loss_class=nn.CrossEntropyLoss,
):
    """Evaluate the model and return average eval_loss and eval_accuracy."""
    model.to(device)
    loss_fn = loss_class().to(device)
    model.eval()
    correct, loss_total, total = 0, 0, 0
    with torch.no_grad():
        for batch in testloader:
            inputs = batch[input_features[0]].to(device)
            targets = batch[target_features[0]].to(device)
            outputs = model(inputs)
            loss_total += loss_fn(outputs, targets).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    eval_accuracy = correct / total if total > 0 else 0
    eval_loss = loss_total / len(testloader) if len(testloader) > 0 else 0
    return eval_loss, eval_accuracy


def get_weights(model: nn.Module, ):
    """Return the model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: nn.Module, parameters):
    """Set the model weights from a list of NumPy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
