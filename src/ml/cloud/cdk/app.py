"""CDK app entrypoint for ML cloud training resources."""

from __future__ import annotations

import os

from aws_cdk import App

from stack import MlCloudTrainingStack


def main() -> None:
    """Instantiate and synthesize the ML cloud training stack."""
    app = App()

    # Get IAM username from context or environment variable
    training_user_name = (
        app.node.try_get_context("training_user_name")
        or os.getenv("TRAINING_USER_NAME")
    )

    MlCloudTrainingStack(
        app,
        "MlCloudTrainingStack",
        training_user_name=training_user_name,
    )
    app.synth()


if __name__ == "__main__":
    main()
