"""
Freeze + Extend: create extended checkpoint files for each input mode.

Takes the pretrained Toronto3D checkpoint (6 channels) and produces new
checkpoint files with extended fc0 weights for BEV / 2D / dual-input modes.

This is optional — main.py does Freeze+Extend on-the-fly at load time.
Use this script if you want to save the extended checkpoints to disk for
faster startup or for fine-tuning.

Usage:
  python extend_weights.py --input weights/randlanet_toronto3d_202201071330utc.pth

Outputs:
  weights/randlanet_toronto3d_bev_11ch.pth      (Mode 2: 3D + BEV)
  weights/randlanet_toronto3d_2d_9ch.pth        (Mode 3: 3D + 2D footprint+DEM)
  weights/randlanet_toronto3d_dual_14ch.pth      (Mode 4: 3D + BEV + 2D)
"""

import os
import sys
import argparse
import torch


PRETRAINED_IN_CHANNELS = 6

# All mode definitions
MODES = {
    "bev": {
        "in_channels": 11,
        "suffix": "bev_11ch",
        "description": "3D + BEV (5 extra channels)",
    },
    "2d": {
        "in_channels": 9,
        "suffix": "2d_9ch",
        "description": "3D + 2D footprint + DEM (3 extra channels)",
    },
    "dual": {
        "in_channels": 14,
        "suffix": "dual_14ch",
        "description": "3D + BEV + 2D footprint + DEM (8 extra channels)",
    },
    "2d_dsm": {
        "in_channels": 10,
        "suffix": "2d_dsm_10ch",
        "description": "3D + 2D footprint + DEM + DSM (4 extra channels)",
    },
    "dual_dsm": {
        "in_channels": 15,
        "suffix": "dual_dsm_15ch",
        "description": "3D + BEV + 2D footprint + DEM + DSM (9 extra channels)",
    },
}


def extend_checkpoint(ckpt_path: str, target_in_channels: int) -> dict:
    """Load and extend pretrained checkpoint for target channel count."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    extra = target_in_channels - PRETRAINED_IN_CHANNELS
    extended_state_dict = {}

    for key, tensor in state_dict.items():
        if tensor.dim() < 2:
            extended_state_dict[key] = tensor
            continue

        last_dim = tensor.shape[-1]

        if last_dim == PRETRAINED_IN_CHANNELS:
            new_shape = list(tensor.shape)
            new_shape[-1] = target_in_channels
            extended = torch.zeros(new_shape, dtype=tensor.dtype)
            extended[..., :PRETRAINED_IN_CHANNELS] = tensor

            fan_in = target_in_channels
            fan_out = tensor.shape[0] if tensor.dim() == 2 else tensor.shape[-2]
            std = (2.0 / (fan_in + fan_out)) ** 0.5
            torch.nn.init.normal_(extended[..., PRETRAINED_IN_CHANNELS:], mean=0.0, std=std)

            extended_state_dict[key] = extended
            print(f"  Extended: {key}  {list(tensor.shape)} → {list(extended.shape)}")

        elif last_dim == 2 * PRETRAINED_IN_CHANNELS:
            new_last = 2 * target_in_channels
            new_shape = list(tensor.shape)
            new_shape[-1] = new_last
            extended = torch.zeros(new_shape, dtype=tensor.dtype)
            extended[..., :2 * PRETRAINED_IN_CHANNELS] = tensor

            fan_in = new_last
            fan_out = tensor.shape[0] if tensor.dim() == 2 else tensor.shape[-2]
            std = (2.0 / (fan_in + fan_out)) ** 0.5
            torch.nn.init.normal_(extended[..., 2 * PRETRAINED_IN_CHANNELS:], mean=0.0, std=std)

            extended_state_dict[key] = extended
            print(f"  Extended: {key}  {list(tensor.shape)} → {list(extended.shape)}")

        else:
            extended_state_dict[key] = tensor

    return extended_state_dict


def main():
    parser = argparse.ArgumentParser(
        description="Extend pretrained Toronto3D checkpoint for extra input channels"
    )
    parser.add_argument("--input", required=True,
                        help="Path to pretrained checkpoint (.pth)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--modes", nargs="*", default=["bev", "2d", "dual"],
                        choices=list(MODES.keys()),
                        help="Which modes to generate (default: bev 2d dual)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Checkpoint not found: {args.input}")

    output_dir = args.output_dir or os.path.dirname(args.input) or "."
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(args.input))[0]

    for mode_name in args.modes:
        mode = MODES[mode_name]
        print(f"\n--- {mode['description']} ---")

        extended_sd = extend_checkpoint(args.input, mode["in_channels"])

        out_path = os.path.join(output_dir, f"{base_name}_{mode['suffix']}.pth")
        torch.save(extended_sd, out_path)
        print(f"  Saved → {out_path}")

    print("\nDone. Use these checkpoints with the matching config:")
    print("  Mode 2 (BEV):   --config randlanet_toronto3d_bev_config.yml")
    print("  Mode 3 (2D):    --config with in_channels=9")
    print("  Mode 4 (dual):  --config randlanet_toronto3d_dual_config.yml")


if __name__ == "__main__":
    main()
