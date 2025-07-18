from ultralytics import YOLO

def get_yolov8_model(
    pretrained: str = 'yolov8s.pt',
    data_yaml: str = 'data.yaml',
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 512,
    lr0: float = 0.001,
    project: str = 'runs/train',
    name: str = 'exp1',
    exist_ok: bool = True
):
    """
    YOLOv8 학습/추론 API 래퍼
    """
    model = YOLO(pretrained)
    # 학습용
    def train():
        return model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            cache=False,
            lr0=lr0,
            project=project,
            name=name,
            exist_ok=exist_ok
        )
    # 추론용
    def predict(source: str, conf: float = 0.25, iou: float = 0.45 , save_dir: str = 'runs/predict'):
        return model.predict(
            source=source,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            save=True,
            project=save_dir,
            name=name,
            exist_ok=exist_ok
        )
    return train, predict