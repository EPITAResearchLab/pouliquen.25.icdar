import hydra
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
import os


@hydra.main(version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    """Entry point to train a model."""
    seed_everything(cfg.seed, workers=True)

    if "model_overrides" in cfg.training.trainer:
        OmegaConf.set_struct(cfg.training.trainer, value=False)
        cfg.training.trainer.model = OmegaConf.merge(cfg.training.trainer.model,
                                           cfg.training.trainer.model_overrides)
        del cfg.training.trainer["model_overrides"]
        OmegaConf.set_struct(cfg.training.trainer, value=True)

    trainer = instantiate(cfg.training.trainer)

    datamodule = instantiate(cfg.training.datamodule)

    data_val = instantiate(cfg.data.train)
    decision = instantiate(cfg.decision)

    model = instantiate(cfg.model)

    trainer.val(model, decision, data_val)

    trainer.train(datamodule, cfg.task_name)


if __name__ == "__main__":
    main()
