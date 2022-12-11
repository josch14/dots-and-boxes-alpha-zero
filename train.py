from lib.trainer import Trainer
import yaml


def main(configuration: dict):
    trainer = Trainer(configuration=configuration)
    trainer.loop()

    # TODO save model
    save_model = trainer.model()


if __name__ == '__main__':
    CONFIG_FILE = "resources/train_config.yaml"

    with open(CONFIG_FILE) as f:
        config = yaml.safe_load(f)

    main(configuration=config)
