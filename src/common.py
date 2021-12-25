from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.transforms import Compose, Normalize, ToTensor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(config_name):
    with open(config_name) as f:
        config = yaml.safe_load(f)
    return config

def load_data(data_root: str, batch_size: int) -> Tuple[DataLoader, DataLoader, Dict[str, str]]:
    """Loading train and test datasets"""
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    num_examples = {"trainset": len(train_dataset), "testset": len(test_dataset)}
    return train_loader, test_loader, num_examples

def build_model(input_size: int, output_size: int, hidden_sizes: list) -> nn.Sequential:
    """Create the neural network model"""
    return nn.Sequential(
                      nn.Linear(input_size, hidden_sizes[0]),
                      nn.Tanh(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.Tanh(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

def train(model, trainloader: DataLoader, optimizer, scheduler, privacy_engine, epochs: int=3, tag="centralized") -> None:
    """Train the network on the training set."""
    config = load_config("project_conf.yaml")
    model.train()
    writer = SummaryWriter(f"runs/{tag}")
    criterion = nn.NLLLoss()
    num_samples = 0
    n_iter = 0
    for epoch in range(epochs):
        correct, total = 0, 0
        running_loss, running_acc = 0, 0
        with BatchMemoryManager(
            data_loader=trainloader,
            max_physical_batch_size=64,
            optimizer=optimizer) as memory_safe_data_loader:
            for i , (image, label) in enumerate(memory_safe_data_loader):
                optimizer.zero_grad()
                image, label = image.to(DEVICE), label.to(DEVICE)
                image = image.view(image.shape[0], -1)
                # forward propagation
                outputs = model(image)
                loss = criterion(outputs, label)
                # backward propagation
                loss.backward()
                # Gardient Descent
                optimizer.step()
                total += label.size(0)
                num_samples += label.size(0)
                correct += (outputs.argmax(1) == label).sum().item()
                running_acc = correct / total
                running_loss += loss.item()
                if i % 100 == 99:
                    writer.add_scalar("Loss/train", running_loss, n_iter)
                    writer.add_scalar("Accuracy/train", running_acc, n_iter)
                    epsilon = privacy_engine.get_epsilon(config["delta"])
                    writer.add_scalar('Privacy Loss/train', epsilon, n_iter)
                    running_acc, running_loss = 0, 0
                    total, correct = 0, 0
                n_iter += 1
            scheduler.step()
            return running_loss, running_acc, num_samples

def test(model, testloader: DataLoader) -> Tuple[float, float, int]:
    """Test the network on the test set"""
    model.eval()
    correct, loss, num_samples = 0, 0.0, 0
    citerion = nn.NLLLoss()
    for image, label in testloader:
        image, label = image.to(DEVICE), label.to(DEVICE)
        image = image.view(image.shape[0], -1)
        outputs = model(image)
        loss += citerion(outputs, label).item()
        num_samples += len(label)
        correct += (outputs.argmax(1) == label).sum().item()
    accuracy = correct / num_samples
    return loss, accuracy, num_samples
