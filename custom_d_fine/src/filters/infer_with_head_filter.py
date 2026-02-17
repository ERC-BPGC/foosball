from pathlib import Path
from shutil import rmtree
import csv

import cv2
import hydra
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from src.infer.torch_model import Torch_model
from src.filters.head_filter import HeadFilter, HeadFilterConfig


def figure_input_type(folder_path: Path):
    video_types = ["mp4", "avi", "mov", "mkv"]
    img_types = ["jpg", "png", "jpeg"]
    data_type = None
    for f in folder_path.iterdir():
        if f.suffix[1:] in video_types:
            data_type = "video"; break
        elif f.suffix[1:] in img_types:
            data_type = "image"; break
    logger.info(f"Head-filter inferencing on data type: {data_type}, path: {folder_path}")
    return data_type


def build_output_roots(root: Path):
    # Three parallel trees
    det = root / "detections"
    dis = root / "distractors"
    hed = root / "head_detections"
    for p in [det, dis, hed]:
        (p / "images").mkdir(parents=True, exist_ok=True)
        (p / "labels").mkdir(parents=True, exist_ok=True)
    return {"detections": det, "distractors": dis, "heads": hed}


def run_on_images(torch_model, folder_path, out_roots, label_to_name, head_filter: HeadFilter, stats):
    img_paths = [p for p in folder_path.iterdir() if p.suffix.lower()[1:] in ("jpg", "jpeg", "png")]
    img_paths.sort()
    batch = 0
    for img_path in tqdm(img_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        res = torch_model(img)
        res = {
            "boxes": res[batch]["boxes"],
            "labels": res[batch]["labels"],
            "scores": res[batch]["scores"],
        }
        head_filter.process_frame(
            img_bgr=img,
            ball_boxes=res["boxes"],
            ball_scores=res["scores"],
            ball_labels=res["labels"],
            out_dirs=out_roots,
            img_name_stem=img_path.stem,
            ball_class_id=0,
            stats_list=stats,
        )

    for k in ["detections", "distractors"]:
        with open(out_roots[k] / "labels.txt", "w") as f:
            f.write("ball\n")
    with open(out_roots["heads"] / "labels.txt", "w") as f:
        f.write("head\n")


def run_on_videos(torch_model, folder_path, out_roots, label_to_name, head_filter: HeadFilter, stats):
    vid_paths = [p for p in folder_path.iterdir() if p.suffix.lower()[1:] in ("mp4", "avi", "mov", "mkv")]
    vid_paths.sort()
    batch = 0
    for vp in tqdm(vid_paths):
        cap = cv2.VideoCapture(str(vp))
        idx = 0
        ok, img = cap.read()
        while ok:
            idx += 1
            frame_name = f"{vp.stem}_frame_{idx}"
            res = torch_model(img)
            res = {
                "boxes": res[batch]["boxes"],
                "labels": res[batch]["labels"],
                "scores": res[batch]["scores"],
            }
            head_filter.process_frame(
                img_bgr=img,
                ball_boxes=res["boxes"],
                ball_scores=res["scores"],
                ball_labels=res["labels"],
                out_dirs=out_roots,
                img_name_stem=frame_name,
                ball_class_id=0,
                stats_list=stats,
            )
            ok, img = cap.read()

    for k in ["detections", "distractors"]:
        with open(out_roots[k] / "labels.txt", "w") as f:
            f.write("ball\n")
    with open(out_roots["heads"] / "labels.txt", "w") as f:
        f.write("head\n")


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig):
    torch_model = Torch_model(
        model_name=cfg.model_name,
        model_path=cfg.train.pretrained_model_path,
        n_outputs=len(cfg.train.label_to_name),
        input_width=cfg.train.img_size[1],
        input_height=cfg.train.img_size[0],
        conf_thresh=cfg.train.conf_thresh,
        rect=cfg.export.dynamic_input,
        half=cfg.export.half,
    )

    hf_cfg = HeadFilterConfig(
        head_model_path=cfg.filter.head_model_path,
        head_conf_threshold=cfg.filter.head_conf_threshold,
        crop_scale=cfg.filter.crop_scale,
        proximity_factor=cfg.filter.proximity_factor,
        min_overlap=cfg.filter.min_overlap,
        save_debug=cfg.filter.save_debug,
        max_heads_to_draw=cfg.filter.max_heads_to_draw,
    )
    head_filter = HeadFilter(hf_cfg, label_to_name=cfg.train.label_to_name)

    folder_path = Path(str(cfg.train.path_to_test_data))
    data_type = figure_input_type(folder_path)

    root_out = Path(cfg.train.root) / "output"
    out_roots = build_output_roots(root_out)

    for k in out_roots:
        for sub in ["images", "labels"]:
            p = out_roots[k] / sub
            if p.exists():
                for f in p.glob("*"):
                    f.unlink()

    stats = []

    if data_type == "image":
        run_on_images(torch_model, folder_path, out_roots, cfg.train.label_to_name, head_filter, stats)
    elif data_type == "video":
        run_on_videos(torch_model, folder_path, out_roots, cfg.train.label_to_name, head_filter, stats)
    else:
        logger.error("Unsupported data type. Provide JPG/PNG images or a video file(s).")
        return

    csv_path = root_out / "frame_stats.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "num_ball_detections", "num_heads_detected", "num_distractors"])
        writer.writerows(stats)

    total_distractors = sum(row[3] for row in stats)
    logger.info(f"Total distractors: {total_distractors}")


if __name__ == "__main__":
    main()
