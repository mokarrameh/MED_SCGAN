r"""FID, KID and IS between a folder of real images and folders of generated images.

Calls torch_fidelity.calculate_metrics(input1=<real folder>, input2=<generated folder>,
isc=True, fid=True, kid=True, kid_subset_size=100). All other settings are torch-fidelity
defaults: feature extractor 'inception-v3-compat' (2048-d pool features; images are resized
inside the extractor to 299 x 299), 100 KID subsets, polynomial kernel of degree 3,
rng_seed 2020. The paper reports FID and KID for WLI -> NBI: --gt is the real NBI test
folder (test/B) and the model folder is the B folder written by a *_test.py script.
The paper does not report IS.

Example:
    python utils/evaluate_models.py --gt datasets/scgan_comp/test/B \
        --models SCGAN_ResNet9=checkpoints/scgan_res9/B --kid_subset_size 100
"""
import argparse
from torch_fidelity import calculate_metrics  # adjust import if needed

def evaluate_model(name, input1, input2, cuda=False, isc=True, fid=True, kid=True, kid_subset_size=100):
    print(f"\n=== Evaluating model: {name} ===")
    metrics = calculate_metrics(
        input1=input1,
        input2=input2,
        cuda=cuda,
        isc=isc,
        fid=fid,
        kid=kid,
        kid_subset_size=kid_subset_size
    )

    # Print results
    for metric_name, value in metrics.items():
        if isinstance(value, (list, tuple)):
            print(f"{metric_name}: {value}")
        else:
            print(f"{metric_name}: {value:.4f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate image translation models with IS, FID, and KID metrics.")
    parser.add_argument("--gt", type=str, required=True,
                        help="Path to ground truth dataset (e.g., testB).")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="List of model outputs in format name=path. Example: "
                             "--models CycleGAN_UNet=path/to/output CycleGAN_ResNet6=path/to/output2")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available.")
    parser.add_argument("--kid_subset_size", type=int, default=100,
                        help="Subset size for KID calculation.")

    args = parser.parse_args()

    # Parse model name=path pairs
    models = {}
    for m in args.models:
        if "=" not in m:
            raise ValueError(f"Model entry '{m}' must be in the format name=path")
        name, path = m.split("=", 1)
        models[name] = path

    results = {}
    for name, out_path in models.items():
        results[name] = evaluate_model(
            name=name,
            input1=args.gt,
            input2=out_path,
            cuda=args.cuda,
            isc=True,
            fid=True,
            kid=True,
            kid_subset_size=args.kid_subset_size
        )

    print("\n=== Summary ===")
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric_name, value in metrics.items():
            if isinstance(value, (list, tuple)):
                print(f"  {metric_name}: {value}")
            else:
                print(f"  {metric_name}: {value:.4f}")
