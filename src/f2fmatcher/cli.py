import argparse
import sys

from f2fmatcher.config import load_config
from f2fmatcher.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="F2FMatcher — Fiber-to-fiber matching across histological stains")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run-pipeline
    p_run = subparsers.add_parser("run-pipeline", help="Run fiber matching for one or more image pairs")
    p_run.add_argument("--param-file", type=str, help="YAML config file for all parameters")
    p_run.add_argument("--img1", type=str, help="Image 1 name (no extension)")
    p_run.add_argument("--img2", type=str, help="Image 2 name (no extension)")
    p_run.add_argument("--source1", type=str, help="Directory of image 1")
    p_run.add_argument("--source2", type=str, help="Directory of image 2")
    p_run.add_argument("--czi1", action="store_true", help="Image 1 is CZI format")
    p_run.add_argument("--czi2", action="store_true", help="Image 2 is CZI format")
    p_run.add_argument("--channel1", type=int, default=1, help="Channel index for image 1")
    p_run.add_argument("--channel2", type=int, default=1, help="Channel index for image 2")
    p_run.add_argument("--cp-model-1", type=str, help="Cellpose model name for image 1")
    p_run.add_argument("--cp-model-2", type=str, help="Cellpose model name for image 2")
    p_run.add_argument("--param-img1", type=str, default="fluorescence", help="fluorescence or brightfield")
    p_run.add_argument("--param-img2", type=str, default="fluorescence", help="fluorescence or brightfield")
    p_run.add_argument("--obj1", type=str, default="10X", help="Objective of image 1")
    p_run.add_argument("--obj2", type=str, default="10X", help="Objective of image 2")
    p_run.add_argument("--output", type=str, required=True, help="Output directory")
    p_run.add_argument("--export-images", action="store_true", help="Export prediction visualization")
    p_run.add_argument("--save-step-prediction", action="store_true", help="Save step-by-step prediction")
    p_run.add_argument("--n-processes", type=int, default=60, help="Number of parallel processes")
    p_run.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    p_run.add_argument("--skip-vae-inputs", action="store_true", help="Skip generating VAE inputs (load precomputed)")
    p_run.add_argument("--skip-embeddings", action="store_true", help="Skip generating embeddings (load precomputed)")

    # train-vae
    p_vae = subparsers.add_parser("train-vae", help="Train the VAE model")
    p_vae.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    p_vae.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    p_vae.add_argument("--device", type=str, default="cuda", help="Device")

    # train-classifier
    p_cls = subparsers.add_parser("train-classifier", help="Train the pair classifier")
    p_cls.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    p_cls.add_argument("--checkpoint-path", type=str, default="./classifier.pth", help="Path to save checkpoint")
    p_cls.add_argument("--device", type=str, default="cuda", help="Device")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run-pipeline":
        from f2fmatcher.scripts.run_pipeline import run_pipeline
        run_pipeline(args)
    elif args.command == "train-vae":
        from f2fmatcher.scripts.train_vae import train_vae_main
        train_vae_main(args)
    elif args.command == "train-classifier":
        from f2fmatcher.scripts.train_classifier import train_classifier_main
        train_classifier_main(args)


if __name__ == "__main__":
    main()
