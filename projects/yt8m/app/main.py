"""Run Flask app for YouTube-8M model demo."""

import os
import sys
import shutil
import tempfile
import subprocess
import base64

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import googleapiclient.discovery
import tensorflow as tf
import numpy as np


app = Flask(__name__)
socketio = SocketIO(app)

PROJECT = os.environ["PROJECT"]
MODEL = "yt8m_video"  # Name of the deployed ai-platform model.
LABEL_VOCAB_FILE = "../data/vocabulary.csv"
VIDEO_DIR = "test_videos"
TFREC_DIR = "test_tfrecords"
YT_DL = "bin/youtube-dl"
FT_EXTRACTOR = "feature_extractor/extract_tfrecords_main.py"


def read_label_vocab(infile=LABEL_VOCAB_FILE):
    with open(infile, "rt") as f:
        raw_vocab = [l.strip("\n") for l in f.readlines()]
    header = raw_vocab[0].split(",")
    index_pos = header.index("Index")
    label_pos = header.index("Name")
    vocab = {}
    for line in raw_vocab[1:]:
        line = line.split(",")
        vocab[int(line[index_pos])] = line[label_pos]
    return vocab


def predict_json(instances, project=PROJECT, model=MODEL, version=None):
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    service = googleapiclient.discovery.build("ml", "v1")
    name = "projects/{}/models/{}".format(project, model)
    if version is not None:
        name += "/versions/{}".format(version)
    response = service.projects().predict(
        name=name,
        body={"instances": instances}
    ).execute()
    if "error" in response:
        raise RuntimeError(response["error"])
    return response["predictions"][0]


def parse_tfrecord(tfrec_file):
    """Encode tfrecord serialized string in base64."""
    rec_iter = tf.io.tf_record_iterator(tfrec_file)
    body = {"b64": base64.b64encode(next(rec_iter)).decode("utf-8")}
    return body


def video_to_tfrecord(video_file):
    video_tag = os.path.basename(video_file).split(".")[0]
    tmpcsv, tmpcsv_name = tempfile.mkstemp()
    tmprec, tmprec_name = tempfile.mkstemp()
    with open(tmpcsv_name, "wt") as f:
        f.write("{},0\n".format(video_file))
    p = subprocess.Popen([
        "python", FT_EXTRACTOR,
        "--input_videos_csv", tmpcsv_name,
        "--output_tfrecords_file", tmprec_name,
        "--skip_frame_level_features", "false"
        ], stdout=sys.stdout)    
    out, err = p.communicate()
    return tmprec_name


def download_yt(video_link, outdir=VIDEO_DIR):
    """Use youtube-dl to download a youtube video.
    https://github.com/ytdl-org/youtube-dl
    """
    video_tag = os.path.basename(video_link)
    outfile = os.path.join(outdir, "{}.mp4".format(video_tag))
    p = subprocess.Popen([
        YT_DL, video_link,
        "-o", outfile,
        "-k",
        "-f", "mp4"
        ], stdout=sys.stdout)    
    out, err = p.communicate()
    return outfile


def inspect_tfrec(tfrec_file, is_sequence=False):
    """Print a tfrecord file content."""
    record_iter = tf.io.tf_record_iterator(tfrec_file)
    if is_sequence:
        example = tf.train.SequenceExample()
        example.ParseFromString(next(record_iter))
    else:
        example = tf.train.Example()
        example.ParseFromString(next(record_iter))
    return example


vocab = read_label_vocab()


@socketio.on("predict_request", namespace="")
def start_predict_pipeline(message):
    # Form iframe to autoplay the requested youtube video.
    video_link = message["link"]
    video_tag = os.path.basename(video_link)
    emit("video_response", {"tag": video_tag})

    # Do prediction.
    # Check if the video is already processed before.
    tfrec_file = os.path.join(TFREC_DIR, "{}.tfrecord".format(video_tag))
    if not os.path.exists(tfrec_file):
        # Download the youtube video as mp4.
        emit("status_update", {"status": "Start Downloading video..."})
        video_file = download_yt(video_link)
        if os.path.exists(video_file):
            emit("status_update", {"status": "Download completed."})
        else:
            emit("status_update", {"status": "Invalid link!"})
            return
        # Convert mp4 to tfrecord.
        emit("status_update", {"status": "Extracting video embeddings..."})
        tmp_tfrec_file = video_to_tfrecord(video_file)
        shutil.move(tmp_tfrec_file, tfrec_file)
        emit("status_update", {"status": "Feature extraction completed."})
    # Request online prediction service.
    emit("status_update", {"status": "Request online predictions..."})
    request_data = parse_tfrecord(tfrec_file)
    responses = predict_json(request_data)
    emit("status_update", {"status": "All done!"})
    # Tidy predictions.
    predictions = {}
    proba = np.array(responses["activation"])
    top_k_pos = proba.argsort()[-10:][::-1]
    predictions["top_k"] = ["{}: {:.2%}".format(vocab[c], p) for c, p in 
        zip(top_k_pos, proba[top_k_pos])]
    predictions["n_class"] = str((proba > .5).sum())
    emit("predict_response", predictions)


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("index.html")


if __name__ == "__main__":
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    socketio.run(app,host="0.0.0.0", port=8080, debug=True)
