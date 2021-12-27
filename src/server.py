import flwr as fl


def main():
    def fit_config(rnd):
        config = {
            "epoch_global": str(rnd),
            "num_epochs": 1,
            "batch_size": 64,
            "optim_lr": 0.001,
            "sched_step_size": 1,
            "sched_gamma": 0.6
        }
        return config

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        on_fit_config_fn=fit_config
    )
    fl.server.start_server(config={"num_rounds": 3},
                           strategy=strategy,
                           force_final_distributed_eval=True)


if __name__ == "__main__":
    main()
