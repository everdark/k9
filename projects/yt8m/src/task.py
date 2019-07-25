"""Model interface for gcloud ai-platform."""

import argparse
import json
import os

from . import model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path",
        help="GCS or local path to training data",
        required = True
    )
    parser.add_argument(
        "--train_steps",
        help="Steps to run the training job for (default: 1000)",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--train_epochs",
        help="Epochss to run the training job for (default: 1)",
        type=int,
        default=1
    )
    parser.add_argument(
        "--eval_data_path",
        help="GCS or local path to evaluation data",
        required= True
    )
    parser.add_argument(
        "--model_dir",
        help="GCS location to write checkpoints and export models",
        required=True
    )
    parser.add_argument(
        "--weighted_loss",
        help = "Use class weights in loss?",
        required=False,
        default=True
    )
    parser.add_argument(
        "--job-dir",
        help="This is not used by our model, but it is required by gcloud",
    )
    args = parser.parse_args().__dict__

    # Append trial_id to path so trials don"t overwrite each other
    args["model_dir"] = os.path.join(
        args["model_dir"],
        json.loads(
            os.environ.get("TF_CONFIG", "{}")
        ).get("task", {}).get("trial", "")
    )

    # Run the training job
    yt8m_model = model.BaseModel(args)
    yt8m_model.train_and_evaluate(args)