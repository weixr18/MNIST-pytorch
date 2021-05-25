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
            "learning_rate": 1e-2,
            "optimizer": "SGD",
            # "adam_betas": (0.9, 0.999),
            "momentum": 0.9,
        }
    },
    "mlp1": {
        "train_params": {
            "batch_size": 64,
            "epochs": 10,
            "epoch_lapse": 1,
            "epoch_save": 20,
            "input_shape": [1, 28*28],
        },
        "hyper_params": {
            "learning_rate": 1e-2,
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
            "learning_rate": 1e-2,
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
            "learning_rate": 1e-2,
            "optimizer": "SGD",
            # "adam_betas": (0.9, 0.999),
            "momentum": 0.9,
        }
    },
    "vit": {
        "train_params": {
            "batch_size": 64,
            "epochs": 10,
            "epoch_lapse": 1,
            "epoch_save": 5,
            "input_shape": [1, 28, 28],
            "p_len": 7,
            "n_patches": 4*4
        },
        "hyper_params": {
            "learning_rate": 1e-2,
            "optimizer": "SGD",
            # "adam_betas": (0.9, 0.999),
            "momentum": 0.9,
        }
    },
    "vfnet": {
        "train_params": {
            "batch_size": 64,
            "epochs": 10,
            "epoch_lapse": 1,
            "epoch_save": 20,
            "input_shape": [1, 28, 28],
        },
        "hyper_params": {
            "learning_rate": 1e-2,
            "optimizer": "SGD",
            # "adam_betas": (0.9, 0.999),
            "momentum": 0.9,
        }
    },
    "mlpmixer": {
        "train_params": {
            "batch_size": 64,
            "epochs": 10,
            "epoch_lapse": 1,
            "epoch_save": 5,
            "input_shape": [1, 28, 28],
            "p_len": 7,
            "n_patches": 4*4,
            "hidden_dim_1": 64,
            "hidden_dim_2": 64,
        },
        "hyper_params": {
            "learning_rate": 1e-2,
            "optimizer": "SGD",
            # "adam_betas": (0.9, 0.999),
            "momentum": 0.9,
        }
    },
}

cifar_10_config = {
    "lenet": {
        "train_params": {
            "batch_size": 64,
            "epochs": 50,
            "epoch_lapse": 5,
            "epoch_save": 25,
            "input_shape": [3, 32, 32],
        },
        "hyper_params": {
            "learning_rate": 1e-3,
            "lr_decay": 0.9,
            "optimizer": "Adam",
            "adam_betas": (0.9, 0.999),
            # "momentum": 0.9,
            "milestones": [20, 40, 60],
        }
    },
    "mlp1": {
        "train_params": {
            "batch_size": 64,
            "epochs": 100,
            "epoch_lapse": 5,
            "epoch_save": 25,
            "input_shape": [1, 3*32*32],
        },
        "hyper_params": {
            "learning_rate": 1e-3,
            "lr_decay": 0.9,
            "optimizer": "Adam",
            "adam_betas": (0.9, 0.999),
            # "momentum": 0.9,
            "milestones": [20, 40, 60],
        }
    },
    "mlp2": {
        "train_params": {
            "batch_size": 64,
            "epochs": 50,
            "epoch_lapse": 5,
            "epoch_save": 25,
            "input_shape": [1, 3*32*32],
        },
        "hyper_params": {
            "learning_rate": 1e-3,
            "lr_decay": 0.9,
            "optimizer": "Adam",
            "adam_betas": (0.9, 0.999),
            # "momentum": 0.9,
            "milestones": [20, 40, 60],
        }
    },
    "mlp3": {
        "train_params": {
            "batch_size": 64,
            "epochs": 50,
            "epoch_lapse": 5,
            "epoch_save": 25,
            "input_shape": [1, 3*32*32],
        },
        "hyper_params": {
            "learning_rate": 1e-3,
            "lr_decay": 0.9,
            "optimizer": "Adam",
            "adam_betas": (0.9, 0.999),
            # "momentum": 0.9,
            "milestones": [20, 40, 60],
        }
    },
    "vit": {
        "train_params": {
            "batch_size": 64,
            "epochs": 50,
            "epoch_lapse": 5,
            "epoch_save": 25,
            "input_shape": [3, 32, 32],
            "p_len": 7,
            "n_patches": 4*4
        },
        "hyper_params": {
            "learning_rate": 1e-3,
            "lr_decay": 0.9,
            "optimizer": "Adam",
            "adam_betas": (0.9, 0.999),
            # "momentum": 0.9,
            "milestones": [20, 40, 60],
        }
    },
    "vfnet": {
        "train_params": {
            "batch_size": 64,
            "epochs": 50,
            "epoch_lapse": 5,
            "epoch_save": 25,
            "input_shape": [3, 32, 32],
        },
        "hyper_params": {
            "learning_rate": 1e-3,
            "lr_decay": 0.9,
            "optimizer": "Adam",
            "adam_betas": (0.9, 0.999),
            # "momentum": 0.9,
            "milestones": [20, 40, 60],
        }
    },
    "mlpmixer": {
        "train_params": {
            "batch_size": 64,
            "epochs": 50,
            "epoch_lapse": 5,
            "epoch_save": 25,
            "input_shape": [3, 32, 32],
            "p_len": 8,
            "n_patches": 4*4,
            "hidden_dim_1": 64,
            "hidden_dim_2": 64,
        },
        "hyper_params": {
            "learning_rate": 1e-3,
            "lr_decay": 0.9,
            "optimizer": "Adam",
            "adam_betas": (0.9, 0.999),
            # "momentum": 0.9,
            "milestones": [20, 40, 60],
        }
    },
}

config = {
    "mnist": mnist_config,
    "cifar-10": cifar_10_config,
}
