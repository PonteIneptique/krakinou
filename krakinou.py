import json
from collections import defaultdict
import dataclasses
from logging import getLogger, CRITICAL
from typing import Optional, List, Dict, Tuple
import click
from kraken.lib.xml import XMLPage
from kraken.blla import segment
from kraken.lib.vgsl import TorchVGSLModel
import PIL.Image
import tabulate
from shapely.geometry import Polygon

logger = getLogger(__name__)
logger.setLevel(CRITICAL)


@dataclasses.dataclass
class Poly:
    label: Optional[str]
    polygon: Polygon

    def tuple(self):
        return self.label, self.polygon


def compute_iou(poly1: Polygon, poly2: Polygon) -> float:
    try:
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return intersection / union
    except:
        # logger.error(f"Unable to perform IoU on {poly2}")
        return 0


def precision_recall_ap(ground_truths: List[Poly], predictions: List[Poly], iou_threshold: float = 0.5):
    gt_by_label = defaultdict(list)
    pred_by_label = defaultdict(list)

    # Reorganize data per Label
    for gt in ground_truths:
        gt_by_label[gt.label].append(gt.polygon)

    for pred in predictions:
        pred_by_label[pred.label].append(pred.polygon)

    # Prepare output dictionaries
    precision: Dict[str, float] = {}
    recall: Dict[str, float] = {}
    ap: Dict[str, float] = {}
    ious: Dict[str, List[float]] = defaultdict(list)
    support: Dict[str, int] = {}
    preds: Dict[str, int] = {}

    for label in set(gt_by_label.keys()).union(pred_by_label.keys()):
        gt_polygons = gt_by_label[label]
        pred_polygons = pred_by_label[label]

        tp = 0
        fp = 0
        fn = 0

        matched_gt = set()
        matched_pred = set()

        for pred_idx, pred_poly in enumerate(pred_polygons):
            matched = False
            for gt_idx, gt_poly in enumerate(gt_polygons):
                iou = compute_iou(pred_poly, gt_poly)
                # Check that we did not already match the current GT Polygon
                # Check that IoU is bigger than threshold (To me, mAP is > and not >=, but needs checking)
                if gt_idx not in matched_gt and iou > iou_threshold:
                    tp += 1
                    matched_gt.add(gt_idx)
                    matched_pred.add(pred_idx)
                    matched = True
                    ious[label].append(iou)
                    break
            # If the GT is not matched, then we say it's a false-positive
            if not matched:
                fp += 1

        # False negatives: no detection of same label polygons
        fn = len(gt_polygons) - len(matched_gt)
        # False positives: polygons in preds that were not matched in GTs
        fp += len(pred_polygons) - len(matched_pred)

        precision[label] = tp / (tp + fp) if tp + fp > 0 else 0
        recall[label] = tp / (tp + fn) if tp + fn > 0 else 0
        ap[label] = precision[label]  # Simplified AP calculation for a single threshold
        support[label] = len(gt_by_label[label])
        preds[label] = len(pred_by_label[label])

    mAP = sum(ap.values()) / len(ap) if ap else 0

    return mAP, precision, recall, support, preds, ious


@click.command()
@click.argument("ground-truth", nargs=-1, type=click.Path(file_okay=True, dir_okay=False))
@click.option("--model", "-m", type=click.Path(file_okay=True, dir_okay=False), default=None)
@click.option("--iou", "-t", multiple=True, type=float, default=(.5, ), help="Threshold for mAP")
@click.option("--output", "-o", help="Output results to json",
              type=click.Path(file_okay=True, dir_okay=False))
@click.option("--verbose", "-v", help="Verbose", is_flag=True)
@click.option("--device", "-d", help="Kraken device", default="cpu")
def krakinou(ground_truth: List[str], model: str, iou: Tuple[float, ...], output: str, verbose: bool, device: str):
    results = {}
    if model:
        model = TorchVGSLModel.load_model(model)
        model.eval()
    for file in ground_truth:
        results[file] = {}
        ground_truth = XMLPage(file)
        full_im = PIL.Image.open(str(ground_truth.imagename))
        prediction = segment(full_im, model=model, device=device)

        gt_polys: List[Poly] = [
            Poly(gt_line.tags.get("type"), Polygon(gt_line.boundary))
            for gt_line in ground_truth.lines.values()
        ]
        pr_polys: List[Poly] = [
            Poly(p_line.tags.get("type"), Polygon(p_line.boundary))
            for p_line in prediction.lines
        ]

        for threshold in iou:
            mAP, pre, rec, sup, prd, ious = precision_recall_ap(gt_polys, pr_polys, iou_threshold=threshold)

            results[file][threshold] = {
                "mAP": mAP,
                "classes": {
                    "support": sup,
                    "found": prd,
                    "precision": pre,
                    "recall": rec,
                    "average IoU": {
                        label: (sum(iou_list)/len(iou_list) if iou_list else 0)
                        for label, iou_list in ious.items()
                    }
                }
            }

    if verbose:
        for file in results:
            print(f"# Results for {file}\n")
            for threshold, scores in results[file].items():
                print(f"## IoU Threshold: {threshold}")
                print(f"mAP: {scores['mAP']:.2f}")
                columns = ["Label", *[el.capitalize() for el in scores["classes"].keys()]]
                metrics = defaultdict(list)
                for metric, values in scores["classes"].items():
                    for label, score in values.items():
                        metrics[label].append(score)
                rows = [
                    [key, *scores]
                    for key, scores in metrics.items()
                ]
                print(tabulate.tabulate(
                    rows,
                    headers=columns,
                    floatfmt=".2f",
                    tablefmt="markdown"
                ))

    if output:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"--> Results written to {output}")

    return results


if __name__ == "__main__":
    krakinou()
