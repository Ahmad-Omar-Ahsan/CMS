from argparse import ArgumentParser
from config_parser import get_config


import tensorflow as tf
import tensorflow_addons as tfa
import wandb
import os 
import yaml
from utils import seed_everything,  get_model, log, load_dataset
from wandb.keras import WandbCallback


def training_pipeline(config):
    """
    Initiates and executes all the steps involved with model training
    Args:
        config (dict) - Dict containing settings for training.
    """
    config["exp"]["save_dir"] = os.path.join(
        config["exp"]["exp_dir"], config["exp"]["exp_name"]
    )
    os.makedirs(config["exp"]["save_dir"], exist_ok=True)


    ######################################
    # save hyperparameters for current run
    ######################################

    config_str = yaml.dump(config)
    print("Using settings:\n", config_str)

    with open(os.path.join(config["exp"]["save_dir"], "settings.txt"), "w+") as f:
        f.write(config_str)


    # Load dataset
    train_dataset, val_dataset, test_dataset = load_dataset(config=config)

    model = get_model(config["hparams"]["model"])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tfa.optimizers.AdamW(
                    learning_rate = config['hparams']['optim']['lr'],
                    weight_decay = config['hparams']['optim']['weight_decay']
                  ))
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config['exp']['save_dir'], histogram_freq=1)
    with tf.device(config['exp']['device']):
        model.fit(
            train_dataset,
            epochs=config['hparams']['n_epochs'],
            validation_data=val_dataset,
            callbacks=[tensorboard_callback, WandbCallback]
        )
    model.save(config['exp']['save_dir'])


def main(args):
    config = get_config(args.conf)
    seed_everything(config["hparams"]["seed"])

    if config["exp"]["wandb"]:
        if config["exp"]["wandb_api_key"] is not None:
            with open(config["exp"]["wandb_api_key"], "r") as f:
                os.environ["WANDB_API_KEY"] = f.read()

        elif os.environ.get("WANDB_API_KEY", False):
            print(f"Found API key from env variable.")

        else:
            wandb.login()

        with wandb.init(
            project=config["exp"]["proj_name"],
            name=config["exp"]["exp_name"],
            config=config["hparams"],
        ):
            training_pipeline(config)

    else:
        training_pipeline(config)



if __name__ == "__main__":
    parser = ArgumentParser("Driver code.")
    parser.add_argument(
        "--conf", type=str, required=True, help="Path to config.yaml file."
    )
    args = parser.parse_args()

    main(args)