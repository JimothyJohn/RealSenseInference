#!/usr/bin/env bash
import device_patches  # Device specific patches for Jetson Nano (needs to be before importing cv2)

import argparse
import cv2
import rsinfer as rsi
from edge_impulse_linux.image import ImageImpulseRunner
from xmlrpc.server import SimpleXMLRPCServer
import urlib

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    "--classes", type=list, default=["person"], help=("List of classes to download")
)
parser.add_argument(
    "--num_samples", type=int, default=100, help="Number of samples to download."
)
parser.add_argument(
    "--export_dir",
    type=str,
    default="/data/",
    help="Folder where to download the images.",
)
parser.add_argument(
    "--model",
    type=str,
    default="./rsinfer/dogs.eim",
    help="Model file",
)
parser.add_argument(
    "--input",
    type=str,
    default="realsense",
    help="Input media",
)
parser.add_argument(
    "--resolution",
    type=int,
    default=480,
    help="Input resolution (square)",
)
parser.add_argument(
    "--fps",
    type=int,
    default=30,
    help="Input frames per second (fps)",
)
parser.add_argument(
    "--outdir",
    type=str,
    default="",
    help="Output file directory",
)


def main(args):
    rs = rsi.RealSense(640, args.resolution, args.fps)
    with ImageImpulseRunner(args.model) as runner:
        try:
            rs.Start()
            model_info = runner.init()
            print(
                'Loaded runner for "'
                + model_info["project"]["owner"]
                + " / "
                + model_info["project"]["name"]
                + '"'
            )
            labels = model_info["model_parameters"]["labels"]

            locations = []
            for frame in range(args.num_samples):
                raw = rs.Capture()
                img = raw[:, 80:, :3][:, :-80]

                if img is None:
                    print("Failed to load image")
                    return

                # get_features_from_image also takes a crop direction arguments in case you don't have square images
                # Crop here instead eventually
                features, cropped = runner.get_features_from_image(img)
                res = runner.classify(features)
                new_img = img.copy()

                print(
                    "Found %d bounding boxes (%d ms.)"
                    % (
                        len(res["result"]["bounding_boxes"]),
                        res["timing"]["dsp"] + res["timing"]["classification"],
                    )
                )
                for bb in res["result"]["bounding_boxes"]:
                    print(
                        "\t%s (%.2f): x=%d y=%d w=%d h=%d"
                        % (
                            bb["label"],
                            bb["value"],
                            bb["x"],
                            bb["y"],
                            bb["width"],
                            bb["height"],
                        )
                    )
                    # Convert 96x96 coords to 640x480 coords
                    new_x = int(bb["x"] / 96 * 480 + 80)
                    new_y = int(bb["y"] / 96 * 480)
                    x, y, z = rs.Locate(new_x, new_y)
                    if x + y + z > 0:
                        locations.append([x, y, z])
                        new_img = rsi.drawText(
                            img, new_x, new_y, x, y, z, target_class="dog"
                        )
                        print(f"\tLocation (xyz): {x}mm x {y}mm x {z}mm")

                if args.outdir != "":
                    cv2.imwrite(
                        "{}/{:05d}.jpg".format(args.outdir, frame),
                        cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR),
                    )

                else:
                    print(f"Nothing detected")

        finally:
            if rs:
                rs.Stop()
            if runner:
                runner.stop()
            print(locations)


def get_next_pose(p):
    assert type(p) is dict
    pose = urlib.poseToList(p)
    print("Received pose: " + str(pose))
    pose = [-0.18, -0.61, 0.23, 0, 3.12, 0.04]
    return urlib.listToPose(pose)


server = SimpleXMLRPCServer(("", 50000), allow_none=True)
server.RequestHandlerClass.protocol_version = "HTTP/1.1"
server.register_function(get_next_pose, "get_next_pose")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

    # print("Listening on port 50000...")
    # server.serve_forever()
