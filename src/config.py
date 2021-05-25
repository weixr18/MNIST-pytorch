mnist_config = {
    "lenet": {
        "train_params": {
            "batch_size": 64,
            "epochs": 10,
            "epoch_lapse": 1,
            "epoch_save": 20,
        },
        "hyper_params": {
            "learning_rate": 1e-4,
            "optimizer": "SGD",
            # "adam_betas": (0.9, 0.999),
            "momentum": 0.9,
        }
    },
    "mlp1": {
        "train_params": {
            "batch_size": 64,
            "epochs": 15,
            "epoch_lapse": 1,
            "epoch_save": 20,
        },
        "hyper_params": {
            "learning_rate": 5e-6,
            "optimizer": "SGD",
            # "adam_betas": (0.9, 0.999),
            "momentum": 0.9,
        }
    },
    "mlp2": {
        "train_params": {
            "batch_size": 64,
            "epochs": 10,
            "epoch_lapse": 1,
            "epoch_save": 20,
        },
        "hyper_params": {
            "learning_rate": 5e-6,
            "optimizer": "SGD",
            # "adam_betas": (0.9, 0.999),
            "momentum": 0.9,
        }
    },
    "mlp3": {
        "train_params": {
            "batch_size": 64,
            "epochs": 10,
            "epoch_lapse": 1,
            "epoch_save": 20,
        },
        "hyper_params": {
            "learning_rate": 5e-6,
            "optimizer": "SGD",
            # "adam_betas": (0.9, 0.999),
            "momentum": 0.9,
        }
    }
}

cifar_10_config = {
    "lenet": {
        "train_params": {
            "batch_size": 64,
            "epochs": 10,
            "epoch_lapse": 1,
            "epoch_save": 20,
        },
        "hyper_params": {
            "learning_rate": 1e-4,
            "optimizer": "SGD",
            # "adam_betas": (0.9, 0.999),
            "momentum": 0.9,
        }
    },
    "mlp1": {
        "train_params": {
            "batch_size": 64,
            "epochs": 15,
            "epoch_lapse": 1,
            "epoch_save": 20,
        },
        "hyper_params": {
            "learning_rate": 5e-6,
            "optimizer": "SGD",
            # "adam_betas": (0.9, 0.999),
            "momentum": 0.9,
        }
    },
    "mlp2": {
        "train_params": {
            "batch_size": 64,
            "epochs": 10,
            "epoch_lapse": 1,
            "epoch_save": 20,
        },
        "hyper_params": {
            "learning_rate": 5e-6,
            "optimizer": "SGD",
            # "adam_betas": (0.9, 0.999),
            "momentum": 0.9,
        }
    },
    "mlp3": {
        "train_params": {
            "batch_size": 64,
            "epochs": 10,
            "epoch_lapse": 1,
            "epoch_save": 20,
        },
        "hyper_params": {
            "learning_rate": 5e-6,
            "optimizer": "SGD",
            # "adam_betas": (0.9, 0.999),
            "momentum": 0.9,
        }
    }
}

config = {
    "mnist": mnist_config,
    "cifar-10": cifar_10_config,
}
