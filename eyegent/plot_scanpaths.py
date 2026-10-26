import json
import os
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ===============================
# TANGO COLOUR PALETTE
# ===============================

COLORS = {"butter": ['#fce94f', '#edd400', '#c4a000'],
          "orange": ['#fcaf3e', '#f57900', '#ce5c00'],
          "chocolate": ['#e9b96e', '#c17d11', '#8f5902'],
          "chameleon": ['#8ae234', '#73d216', '#4e9a06'],
          "skyblue": ['#729fcf', '#3465a4', '#204a87'],
          "plum": ['#ad7fa8', '#75507b', '#5c3566'],
          "scarletred": ['#ef2929', '#cc0000', '#a40000'],
          "aluminium": ['#eeeeec', '#d3d7cf', '#babdb6',
                        '#888a85', '#555753', '#2e3436'],
          }


def draw_scanpath(fix_x, fix_y, fix_d, alpha=1, invert_y=False, ydim=None):
    if fix_d is None:
        fix_d = np.ones(len(fix_x)) * 1000.

    if invert_y:
        if ydim is None:
            raise RuntimeError('ydim must be provided')
        fix_y = ydim - 1 - np.array(fix_y)

    # Arrows between fixations
    for i in range(1, len(fix_x)):
        plt.arrow(fix_x[i - 1],
                  fix_y[i - 1],
                  fix_x[i] - fix_x[i - 1],
                  fix_y[i] - fix_y[i - 1],
                  alpha=alpha,
                  fc=COLORS['chameleon'][0],
                  ec=COLORS['chameleon'][0],
                  fill=True,
                  shape='full',
                  width=3,
                  head_width=0,
                  head_starts_at_zero=False,
                  overhang=0)

    # Fixation circles
    for i in range(len(fix_x)):
        if i == 0:
            plt.plot(fix_x[i], fix_y[i],
                     marker='o',
                     ms=fix_d[i] / 10,
                     mfc=COLORS['skyblue'][0],
                     mec='black',
                     alpha=0.7)
        elif i == len(fix_x) - 1:
            plt.plot(fix_x[i], fix_y[i],
                     marker='o',
                     ms=fix_d[i] / 10,
                     mfc=COLORS['scarletred'][0],
                     mec='black',
                     alpha=0.7)
        else:
            plt.plot(fix_x[i], fix_y[i],
                     marker='o',
                     ms=fix_d[i] / 10,
                     mfc=COLORS['aluminium'][0],
                     mec='black',
                     alpha=0.7)

    # Fixation order numbers
    for i in range(len(fix_x)):
        plt.text(fix_x[i] - 4,
                 fix_y[i] + 1,
                 str(i + 1),
                 color='black',
                 ha='left',
                 va='center',
                 multialignment='center',
                 alpha=alpha)


def save_scanpath(image_path, query, result, output):
    """Draw a single simulation's scanpath on its source image."""
    scanpath = np.asarray(result.scanpath)
    durations = np.asarray(result.durations)
    if scanpath.ndim != 2 or scanpath.shape[1] != 2 or len(scanpath) == 0:
        raise RuntimeError("The agent produced no fixations. Check the VLM connection and tool output.")
    if len(scanpath) != len(durations):
        raise RuntimeError("The returned scanpath and duration arrays have different lengths")

    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        figure, ax = plt.subplots(figsize=(10, 7))
        try:
            ax.imshow(image)
            plt.sca(ax)
            # The original renderer sizes markers using durations in milliseconds.
            draw_scanpath(scanpath[:, 0], scanpath[:, 1], durations * 1000)
            ax.set_title(query, parse_math=False)
            ax.set_xlim(-0.5, image.width - 0.5)
            ax.set_ylim(image.height - 0.5, -0.5)
            ax.axis("off")
            figure.savefig(output, dpi=180, bbox_inches="tight")
        finally:
            plt.close(figure)
    return output.resolve()


# ===============================
# MAIN FUNCTION
# ===============================

def plot_scanpaths_on_image(image_path, json_files, scanpath_idx=0,
                            output="simulated_scans.jpg"):
    """
    image_path : path to single image
    json_files : list of json file paths
    scanpath_idx : which scanpath to select
    """

    # Load image once
    img = mpimg.imread(image_path)
    img_height = img.shape[0]
    orig_height, orig_width = img.shape[:2]

    n_files = len(json_files)

    fig, axes = plt.subplots(1, n_files, figsize=(6 * n_files, 6))

    # If only one subplot, make axes iterable
    if n_files == 1:
        axes = [axes]

    for ax, json_file in zip(axes, json_files):

        with open(json_file, 'r') as f:
            data = json.load(f)

        # Get first image ID in json
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        questions = data["data"][image_id]

        # Get first question (or modify if needed)
        question = list(questions.keys())[0]
        scanpaths = questions[question]

        if "neva" in json_file:
            scanpath = scanpaths[0]
        elif scanpath_idx >= len(scanpaths):
            raise IndexError(f"Scanpath index {scanpath_idx} out of range in {json_file}")
        else:
            scanpath = scanpaths[scanpath_idx]

        fix_x = np.array(scanpath["x"])
        fix_y = np.array(scanpath["y"])
        fix_d = np.array(scanpath["duration"])

        # Set current axis
        plt.sca(ax)
        ax.imshow(img)
        ax.axis('off')

        if "clipseg" in json_file:

            if np.any(fix_x >= orig_width) or np.any(fix_y >= orig_height):
                raise ValueError(f"Fixation coordinates exceed image bounds in {json_file}")

            # Draw scanpath
            draw_scanpath(fix_x, fix_y, fix_d*1000.,
                        alpha=1,
                        invert_y=False,
                        ydim=img_height)
        else:
            # Draw scanpath
            draw_scanpath(fix_x, fix_y, fix_d,
                        alpha=1,
                        invert_y=False,
                        ydim=img_height)

        ax.set_title(question, fontsize=10)

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return Path(output).resolve()


# ===============================
# COMMAND LINE USAGE
# ===============================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare scanpath JSON files on an image")
    parser.add_argument("image", help="Image used for the scanpaths")
    parser.add_argument("json_files", nargs="+", help="Scanpath JSON files")
    parser.add_argument("--scanpath-idx", type=int, default=0)
    parser.add_argument("--output", default="simulated_scans.jpg")
    args = parser.parse_args()
    print(plot_scanpaths_on_image(args.image, args.json_files,
                                  args.scanpath_idx, args.output))
