"""
Command-line entry point for:

    python -m src.OOD <command>

Examples:

    python -m src.OOD verify
    python -m src.OOD predict path/to/image.png --method combined
"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="MedTrust Out-of-Distribution Detection"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # -----------------------------------------------------
    # VERIFY
    # -----------------------------------------------------

    subparsers.add_parser(
        "verify",
        help="Verify the frozen MedTrust model and OOD configuration.",
    )

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    predict_parser = subparsers.add_parser(
        "predict",
        help="Run OOD inference on one image.",
    )

    predict_parser.add_argument(
        "image",
        type=str,
        help="Path to the input image.",
    )

    predict_parser.add_argument(
        "--method",
        type=str,
        default="combined",
        choices=[
            "msp",
            "energy",
            "mahalanobis",
            "combined",
        ],
        help="OOD scoring method (default: combined).",
    )

    args = parser.parse_args()

    if args.command == "verify":
        from .verify import main as verify_main

        verify_main()

    elif args.command == "predict":
        from .predict import predict_single_image, print_result

        result = predict_single_image(
            image_path=args.image,
            method=args.method,
        )

        print_result(result)


if __name__ == "__main__":
    main()
