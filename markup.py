import pickle
import json

from cvat import CvatDataset
from pycocotools.mask import encode
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class VisionMarkup:
    """Внутреннее представление разметки для задач компьютерного зрения

    На данный момент работает с задачами object detection,
    semantic/instance segmentation.
    Используется в качестве промежуточного формата при конвертации
    между распространенными форматами: CVAT, COCO, ... а также в качестве
    промежуточного формата при конвертации файла с предсказаниями сети
    в нужный для других инструментов формат:
        - CVAT: визуаливация
        - COCO: расчет метрик через cocotools
        - ...
    """

    def __init__(self):
        # соответствие между именем класса объекта и индексом вида {label: id}
        self.object_label_ids = {}

        # данные по каждому изображению, список словарей вида:
        # {
        #     "id": ...,
        #     "name": ...,
        #     "objects": [
        #         {
        #             "id": ...,
        #             "label": ...
        #             "bbox": {"data": [xtl, ytl, xbr, ybr], ["confidence": ...]}
        #             "mask": {
        #                 "format": one of ["rle", "polygons", "array"]
        #                 "data": ...
        #                 ["confidence": ...]
        #             }
        #         },
        #         ...
        #     ],
        #     ...
        # }
        # структура словаря в целом произвольна, обязательно только id изображения, но нужно следить за
        # согласованностью методов load/dump для разных форматов, и, если какие либо поля отсутствуют в
        # одном формате, но обязательны в другом, то нужно позаботиться об их формировани
        self.image_data = []

    @staticmethod
    def load(path):
        """Загрузка из фаила, полученного с помощью dump

        Аргументы:
            path: str
                путь для загрузки

        См. также: dump
        """
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        return loaded

    def dump(self, path):
        """Полное сохранение объекта разметки в машиночитаемом формате

        Аргументы:
            path: str
                путь для сохранения

        См. также: load
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load_cvat(self, path):
        """
        {
            "id", ["name"], "objects": {"id", "label", ["bbox"], ["mask"]}
        }
        CVAT разметка не предоставляет id объектов, поэтому они генерируются,
        при этом считается что каждый элемент внутри <image /> является отдельным объектом
        индексация начинается с 1 (это требование COCO разметки)
        """
        self.image_data = []
        labels = []

        ds = CvatDataset()
        ds.load(path)

        for image_id in ds.get_image_ids():
            record = {"id": image_id}
            if ds.has_name(image_id):
                record["name"] = ds.get_name(image_id)

            if ds.has_size(image_id):
                size = ds.get_size(image_id)
                record["width"] = size["width"]
                record["height"] = size["height"]

            objects = []
            for box in ds.get_boxes(image_id):
                obj = {
                    "label": box["label"], "id": len(objects) + 1,
                    "bbox": {"data": [box["xtl"], box["ytl"], box["xbr"], box["ybr"]]},
                }
                if "conf" in box and box["conf"] is not None:
                    obj["bbox"]["confidence"] = box["conf"]
                objects.append(obj)
                labels.append(box["label"])
            for polygon in ds.get_polygons(image_id):
                obj = {
                    "label": polygon["label"], "id": len(objects) + 1,
                    "mask": {"format": "polygon", "data": polygon["points"]},
                }
                if "conf" in polygon and polygon["conf"] is not None:
                    obj["mask"]["confidence"] = polygon["conf"]
                objects.append(obj)
                labels.append(polygon["label"])
            record["objects"] = objects

            self.image_data.append(record)

        # cvat формат не содержит id классов, только имена, поэтому инициализируем
        # это соответствие сами, начиная с 1. Его можно переопределить в load_cat_ids
        # cvat может не содержать имен классов, тогда они выделются из разметки объектов
        labels = ds.get_labels() or labels
        self.object_label_ids = {label: i + 1 for i, label in enumerate(sorted(labels))}

    def dump_cvat(self, path):
        ds = CvatDataset()
        for record in self.image_data:
            if "id" in record:
                image_id = ds.add_image(record["id"])
            else:
                image_id = ds.add_image()

            if "name" in record:
                ds.set_name(image_id, record["name"])

            if "width" in record and "height" in record:
                ds.set_size(image_id, record["width"], record["height"])

            for obj in record["objects"]:
                label = obj["label"]
                occluded = False
                if "occluded" in obj:
                    occluded = obj["occluded"]
                if "bbox" in obj:
                    bbox = obj["bbox"]
                    xtl, ytl, xbr, ybr = bbox["data"]
                    confidence = None
                    if "confidence" in bbox:
                        confidence = bbox["confidence"]
                    ds.add_box(image_id, xtl, ytl, xbr, ybr, label, occluded, confidence)
                if "mask" in obj:
                    mask = obj["mask"]
                    if mask["format"] == "rle":
                        raise NotImplementedError
                    elif mask["format"] == "polygons":
                        ds.add_polygon(image_id, mask["data"], label, occluded)
                    elif mask["format"] == "array":
                        raise NotImplementedError
                    else:
                        raise NotImplementedError

        ds.dump(path)

    def load_coco(self, path, mode="gt"):
        """mode: ["rt", "gt"]
        """
        raise NotImplementedError

    def dump_coco(self, path, mode="gt"):
        """mode: ["rt", "gt"]
        """
        if mode == "gt":
            images = []
            annotations = []
            categories = [{"id": id_, "name": label, "supercategory": "entity"}
                          for label, id_ in self.object_label_ids.items()]

            for record in self.image_data:
                image_id = record["id"]

                image = {"id": image_id}
                # первые два поля могут быть импортированы из других форматов
                # остальные поля только из COCO фаила
                for k in ["width", "height", "license", "flickr_url", "coco_url", "date_captured"]:
                    if k in record:
                        image[k] = record[k]
                # это поле имеет разное имя
                if "name" in record:
                    image["file_name"] = record["name"]
                images.append(image)

                for obj in record["objects"]:
                    annotation = {
                        "id": obj["id"], "image_id": image_id,
                        "category_id": self.object_label_ids[obj["label"]]
                    }
                    if "bbox" in obj:
                        bbox = obj["bbox"]
                        xtl, ytl, xbr, ybr = bbox["data"]
                        annotation["bbox"] = [xtl, ytl, xbr - xtl, ybr - ytl]
                        annotation["area"] = (xbr - xtl) * (ybr - ytl)
                    if "mask" in obj:
                        mask = obj["mask"]
                        if mask["format"] == "array":
                            segmentation = encode(mask["data"])
                        else:
                            segmentation = mask["data"]
                        annotation["segmentation"] = segmentation
                    annotation["iscrowd"] = obj.get("iscrowd", 0)
                    annotations.append(annotation)

            coco = {"annotations": annotations, "images": images, "categories": categories}

        elif mode == "rt":
            coco = []
            for record in self.image_data:
                for obj in record["objects"]:
                    coco_record = {
                        "image_id": record["id"],
                        "category_id": self.object_label_ids[obj["label"]],
                    }
                    if "mask" in obj:
                        mask = obj["mask"]
                        if mask["format"] == "rle":
                            segmentation = mask["data"]
                        elif mask["format"] == "polygons":
                            raise NotImplementedError
                        elif mask["format"] == "array":
                            segmentation = encode(mask["data"])
                        else:
                            raise Exception(mask["format"] + " not supported")
                        coco_record["segmentation"] = segmentation
                        coco_record["score"] = mask.get("confidence", 1.0)
                    elif "bbox" in obj:
                        bbox = obj["bbox"]
                        xtl, ytl, xbr, ybr = bbox["data"]
                        coco_record["bbox"] = [xtl, ytl, xbr - xtl, ybr - ytl]
                        coco_record["score"] = bbox.get("confidence", 1.0)
                    coco.append(coco_record)
        else:
            raise Exception("Not supported")
        with open(path, "w") as f:
            json.dump(coco, f, indent=2)

    def load_image_ids(self, path):
        """Переопределить соответствие между именем изображения и его id
        Фаил соответствия - csv таблица формата id,name без заголовка

        Аргументы:
            path: str
                путь к фаилу соответствия

        См. также: dump_image_ids
        """
        dct = VisionMarkup._load_match(path)
        dct_inv = {v: k for k, v in dct.items()}
        raise NotImplementedError

    def dump_image_ids(self, path):
        """Сохранить соответствие между именем изображения и его id
        Фаил соответствия - csv таблица формата id,name без заголовка

        Полезно для переноса соответствия из другого формата, либо если
        соответствие хранится в отдельном фаиле

        Аргументы:
            path: str
                путь к фаилу соответствия

        См. также: load_image_ids
        """
        dct = {}
        for record in self.image_data:
            id_ = record["id"]
            name = record["name"]
            dct[name] = id_
        VisionMarkup._dump_match(dct, path)

    def load_label_ids(self, path):
        """Переопределить соответствие между именем класса и его id
        Фаил соответствия - csv таблица формата id,label без заголовка

        Полезно для переноса соответствия из другого формата, либо если
        соответствие хранится в отдельном фаиле

        Аргументы:
            path: str
                путь к фаилу соответствия

        См. также: dump_cat_ids
        """
        self.object_label_ids = VisionMarkup._load_match(path)

    def dump_label_ids(self, path):
        """Сохранить соответствие между именем изображения и его id
        Фаил соответствия - csv таблица формата id,label без заголовка

        Полезно для переноса соответствия из другого формата, либо если
        соответствие хранится в отдельном фаиле

        Аргументы:
            path: str
                путь к фаилу соответствия

        См. также: load_cat_ids
        """
        VisionMarkup._dump_match(self.object_label_ids, path)

    @staticmethod
    def _load_match(path):
        dct = {}
        with open(path, "r") as f:
            for line in f:
                id_, value = line.strip().split(",")
                dct[value] = int(id_)
        return dct

    @staticmethod
    def _dump_match(dct, path):
        with open(path, "w") as f:
            for value, id_ in dct.items():
                f.write("{},{}\n".format(id_, value))

    def merge(self, other):
        """Объединить два объекта разметки

        Полезно при объединении датасетов

        Метод ведет себя аналогично left join из SQL

        Аргументы:
            other: Markup
                объект, который будет добавлен к текущему
        """
        raise NotImplementedError


if __name__ == "__main__":
    vm = VisionMarkup()
    vm.object_label_ids = {"car": 1, "bus": 2}
    vm.image_data.append({"id": 0, "width": 1280, "height": 720, "objects": [
        {"id": 1, "label": "car", "bbox": {"data": [0, 0, 100, 100], "confidence": 1.0}},
        {"id": 2, "label": "bus", "bbox": {"data": [200, 200, 400, 400], "confidence": 1.0}}
    ]})

    vm.dump_cvat("output/cvat_gt.xml")
    # vm.dump_coco("output/coco_rt.json", mode="rt")
    # vm.dump_coco("output/coco_gt.json", mode="gt")
    # vm.dump_label_ids("output/label_ids.csv")
    # vm.dump_image_ids("image_ids.csv")

    # vm.load_cvat("output/cvat_gt.xml")
    # vm.load_label_ids("output/label_ids.csv")

    # vm.dump_coco("output/coco_rt.json", mode="rt")
    # vm.dump_coco("output/coco_gt.json", mode="gt")

    # cocoGt = COCO("output/coco_gt.json")
    # cocoRt = cocoGt.loadRes("output/coco_rt.json")
    #
    # cocoEval = COCOeval(cocoGt, cocoRt, "bbox")
    # cocoEval.evaluate()
    # cocoEval.accumulate()
    # cocoEval.summarize()
