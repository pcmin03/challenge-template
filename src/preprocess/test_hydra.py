from omegaconf import OmegaConf
import hydra
df_config_path = '/opt/rsna/configs/data/sign.yaml'
dataset_config = OmegaConf.load(df_config_path)
hydra.utils.instantiate(dataset_config)