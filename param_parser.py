"""Parses the parameters."""

import argparse
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def parameter_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=device, help="")
    # Data
    parser.add_argument(
        "--data-adj-mat",
        type=str,
        default="data/processed/adj_mat.csv",
        help="Graph adjacency matrix path",
    )
    parser.add_argument(
        "--data-node-feats",
        type=str,
        default="data/processed/node_features.csv",
        help="Graph node features path",
    )
    parser.add_argument(
        "--data-train",
        type=str,
        default="data/processed/NYC_train.csv",
        help="Training data path",
    )
    parser.add_argument(
        "--data-val",
        type=str,
        default="data/processed/NYC_val.csv",
        help="Validation data path",
    )
    parser.add_argument(
        "--data-test",
        type=str,
        default="data/processed/NYC_test.csv",
        help="Testing data path",
    )
    parser.add_argument(
        "--short-traj-thres", type=int, default=2, help="Remove over-short trajectory"
    )
    parser.add_argument(
        "--time-units", type=int, default=48, help="Time unit is 0.5 hour, 24/0.5=48"
    )
    parser.add_argument(
        "--time-feature",
        type=str,
        default="norm_in_day_time",
        help="The name of time feature in the data",
    )
    parser.add_argument(
        "--season-feature",
        type=str,
        default="month_of_year",
        help="The name of season feature in the data",
    )

    # Model hyper-parameters
    parser.add_argument(
        "--poi-embed-dim", type=int, default=128, help="POI embedding dimensions"
    )
    parser.add_argument(
        "--user-embed-dim", type=int, default=128, help="User embedding dimensions"
    )
    parser.add_argument(
        "--gcn-dropout", type=float, default=0.3, help="Dropout rate for gcn"
    )
    parser.add_argument(
        "--gcn-nhid",
        type=list,
        default=[32, 64],
        help="List of hidden dims for gcn layers",
    )
    parser.add_argument(
        "--transformer-nhid",
        type=int,
        default=1024,
        help="Hid dim in TransformerEncoder",
    )
    parser.add_argument(
        "--transformer-nlayers",
        type=int,
        default=2,
        help="Num of TransformerEncoderLayer",
    )
    parser.add_argument(
        "--transformer-nhead",
        type=int,
        default=2,
        help="Num of heads in multiheadattention",
    )
    parser.add_argument(
        "--transformer-dropout",
        type=float,
        default=0.3,
        help="Dropout rate for transformer",
    )
    parser.add_argument(
        "--time-embed-dim", type=int, default=32, help="Time embedding dimensions"
    )
    parser.add_argument(
        "--season-embed-dim", type=int, default=128, help="Time embedding dimensions"
    )
    parser.add_argument(
        "--cat-embed-dim", type=int, default=32, help="Category embedding dimensions"
    )
    parser.add_argument(
        "--time-loss-weight",
        type=int,
        default=10,
        help="Scale factor for the time loss term",
    )
    parser.add_argument(
        "--node-attn-nhid",
        type=int,
        default=128,
        help="Node attn map hidden dimensions",
    )

    # Training hyper-parameters
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train."
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Initial learning rate."
    )
    parser.add_argument(
        "--lr-scheduler-factor",
        type=float,
        default=0.1,
        help="Learning rate scheduler factor",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay (L2 loss on parameters).",
    )

    # Experiment config
    parser.add_argument(
        "--save-weights",
        action="store_true",
        default=True,
        help="whether save the model",
    )
    parser.add_argument(
        "--save-embeds",
        action="store_true",
        default=False,
        help="whether save the embeddings",
    )
    parser.add_argument(
        "--workers", type=int, default=0, help="Num of workers for dataloader."
    )
    parser.add_argument("--project", default="runs", help="save to project/name")
    parser.add_argument("--name", default="iteration", help="save to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disables CUDA training."
    )

    return parser.parse_args()
