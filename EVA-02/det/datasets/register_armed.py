from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# Update these paths to your dataset locations.
register_coco_instances(
    "armed_train",
    {},
    "/kaggle/working/OriginalDataset/annotations/armed_train.json",
    "/kaggle/working/OriginalDataset",
)
register_coco_instances(
    "armed_val",
    {},
    "/kaggle/working/OriginalDataset/annotations/armed_val.json",
    "/kaggle/working/OriginalDataset",
)

MetadataCatalog.get("armed_train").thing_classes = ["Armed", "Unarmed", "Gun"]
MetadataCatalog.get("armed_val").thing_classes = ["Armed", "Unarmed", "Gun"]

