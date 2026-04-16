import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Limit this first run to a single visible GPU so EasyTorch sees one device.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

from basicts import BasicTSLauncher
from basicts.configs import BasicTSForecastingConfig
from basicts.models.StemGNN import StemGNN, StemGNNConfig


def build_config():
    dataset_name = "PEMS08"
    input_len = 12
    output_len = 12
    ckpt_root = "checkpoints/first_run_stemgnn"

    model_config = StemGNNConfig(
        input_len=input_len,
        output_len=output_len,
        num_features=170,
        num_blocks=2,
        hidden_size=8,
        dropout=0.3,
    )

    return BasicTSForecastingConfig(
        model=StemGNN,
        model_config=model_config,
        dataset_name=dataset_name,
        input_len=input_len,
        output_len=output_len,
        use_timestamps=True,
        batch_size=16,
        num_epochs=20,
        lr=0.001,
        gpus="0" if torch.cuda.is_available() else None,
        save_results=True,
        ckpt_save_dir=ckpt_root,
        dataset_params={
            "input_len": input_len,
            "output_len": output_len,
            "use_timestamps": True,
            "memmap": False,
            "data_file_path": os.path.join("datasets", "datasets", dataset_name),
        },
    )


def main():
    cfg = build_config()
    BasicTSLauncher.launch_training(cfg)

    best_ckpt = os.path.join(
        cfg.ckpt_save_dir,
        cfg.md5,
        f"{cfg.model.__name__}_best_val_{cfg.target_metric}.pt",
    )

    if os.path.exists(best_ckpt):
        BasicTSLauncher.launch_evaluation(cfg, best_ckpt, gpus=cfg.gpus, batch_size=cfg.test_batch_size)
    else:
        print(f"Best checkpoint not found: {best_ckpt}")


if __name__ == "__main__":
    main()
