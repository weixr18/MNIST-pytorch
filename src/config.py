mnist_config = {
    "lenet": {
        "train_params": {
            "batch_size": 64,
            "epochs": 10,
            "epoch_lapse": 1,
            "epoch_save": 20,
            "input_shape": [1, 28, 28],
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
            "input_shape": [1, 28*28],
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
            "input_shape": [1, 28*28],
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
            "input_shape": [1, 28*28],
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
            "epochs": 40,
            "epoch_lapse": 5,
            "epoch_save": 4000,
            "input_shape": [3, 32, 32],
        },
        "hyper_params": {
            "learning_rate": 1e-5,
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
            "input_shape": [1, 3*32*32],
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
            "input_shape": [1, 3*32*32],
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
            "input_shape": [1, 3*32*32],
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