import argparse
from collections import OrderedDict

import flwr as fl
import opacus
import torch
from opacus import privacy_engine
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR

from common import DEVICE, build_model, load_config, load_data, test, train


class MNISTClient(fl.client.NumPyClient):
    def __init__(self, cid, model, optimizer, privacy_engine, trainloader, testloader):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.privacy_engine = privacy_engine

    def get_parameters(self):
        return [val.to(DEVICE).numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        scheduler = StepLR(self.optimizer, step_size=config["sched_step_size"], gamma=config["sched_gamma"])
        running_loss, running_acc, num_samples = train(
            self.model, self.trainloader, self.optimizer, scheduler, self.privacy_engine, epochs=config["num_epochs"], tag=f"client_{self.cid}")
        return self.get_parameters(), num_samples, {"running_loss": running_loss, "running_acc": running_acc}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, num_samples = test(self.model, self.testloader)
        return float(loss), num_samples, {"accuracy": float(accuracy)}

def main():
    parser = argparse.ArgumentParser(description="launches clients for MNIST")
    parser.add_argument(
        "--cid",
        type=int,
        required=True,
        help="Define Client_ID"
    )
    args = parser.parse_args()
    # load config file
    config = load_config("project_conf.yaml")
    # Load model
    model = build_model(config["input_size"], config["output_size"], config["hidden_sizes"]).to(DEVICE)
    # load data, we are training locally because FedAvg was proven to not converge on non-IID datasets.
    trainloader, testloader, _ = load_data(config["data_root"], config["batch_size"])
    optimizer = Adam(params=model.parameters(), lr=config["lr"])
    # privacy engine
    privacy_engine = opacus.PrivacyEngine()
    model, optimizer, trainloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        epochs=1,
        target_epsilon=config["epsilon"],
        target_delta=config["delta"],
        max_grad_norm=config["max_grad_norm"],
    )
    client = MNISTClient(args.cid, model, optimizer, privacy_engine, trainloader, testloader)
    fl.client.start_numpy_client("[::]:8080", client=client)

if __name__ == "__main__":
    main()
