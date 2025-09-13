import logging
import os
from pathlib import Path

import hydra
import mlflow
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig, open_dict

from src.utils.utils import already_run, get_best_run, get_metrics, values_as_list

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    """Entry point for calibration."""
    seed_everything(cfg.seed, workers=True)
    data_val = instantiate(cfg.data.train)

    decision = instantiate(cfg.decision)
    if cfg.get("training", ""):
        log.info(cfg.training)
        run_name = cfg.training.trainer.get("run_name", None)
        run_name = f"{cfg.task_name}_{run_name}" if run_name is not None else None
        log.info("%s run name: %s", cfg.task_name, run_name)
        run = get_best_run("lightning_logs", "best_val_loss", cfg.task_name,
                           run_name=run_name)
        if run is not None:
            path = str(Path(run.info.artifact_uri.replace("file://", "")).parent
                    / "checkpoints")
            model_name = next((p for p in os.listdir(path) if p.startswith("backbone_"))
                              , "")
            if len(model_name) > 0:
                with open_dict(cfg):
                    cfg.model.model.model_path = str(Path(path) / model_name)
                log.info("found the model %s", model_name)
            else:
                log.error("couldnt find the network")
                return
        else:
            possible_path = f"bestmodels/{cfg.task_name}"
            if Path(possible_path + ".pth").exists():
                with open_dict(cfg):
                    cfg.model.model.model_path = possible_path + ".pth"
            elif Path(possible_path + ".joblib").exists():
                log.info("joblib exists")
                with open_dict(cfg):
                    cfg.model.model_path = possible_path + ".joblib"
            else:
                log.error("couldnt find the run for the network")
                return
        log.info(cfg.model)
    log.info(cfg.model)
    model = instantiate(cfg.model)
    task_name = cfg.task_name

    task_name_full = HydraConfig.get().runtime.choices.decision + "_" + task_name
    if cfg.get("training", ""):
        task_name_full += cfg.training.trainer.run_name
    mlflow.set_experiment(task_name_full)
    if cfg.get("tune", ""):
        log.info("tunning parameters")
        metrics, th = decision.tune(data_val, model)
        log.info("best fscore found %f for th %s", metrics["fscore"], th)
        cfg.decision.th = float(th)
    else:
        log.info("running for current parameters %s", cfg.decision.th)
        metrics = get_metrics(data_val, model, decision)
        log.info("fscore of %s", metrics["fscore"])
    with mlflow.start_run():
        tag = f"{cfg.get('task_name', '')}_\
            {'_'.join(values_as_list(cfg.model))}_\
            {'_'.join(values_as_list(cfg.decision))}P"
        mlflow.set_tag("mlflow.runName", tag)
        mlflow.log_params(cfg)
        mlflow.log_metrics(metrics)

if __name__ == "__main__":
    main()
