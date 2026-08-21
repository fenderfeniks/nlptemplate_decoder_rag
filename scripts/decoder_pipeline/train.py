import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.endpoints.train import run_universal_train
from src.pipelines.decoder.training.builder import build_decoder_module
from src.utils.cli import enforce_pipeline
from src.utils.hydra_utils import setup_config

load_dotenv()

@hydra.main(config_path="../../configs", config_name="train_decoder", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = setup_config(cfg)
    run_universal_train(
        cfg=cfg, 
        pipeline_name="decoder_pipeline", 
        build_module_fn=build_decoder_module
    )

if __name__ == "__main__":
    enforce_pipeline("decoder_pipeline")
    main()