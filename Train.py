from ultralytics import YOLO

def fine_tune_yolov10(model_config: str, data_config: str, epochs: int, img_size: int) -> None:
    """
    Fine-tune the YOLOv10 model.

    Args:
        model_config (str): Path to the YOLOv10 model configuration file.
        data_config (str): Path to the dataset configuration file.
        epochs (int): Number of epochs for training.
        img_size (int): Image size for training.

    Returns:
        None
    """
    # Load YOLOv10n model from scratch
    model = YOLO(model_config)

    # Train the model
    model.train(data=data_config, epochs=epochs, imgsz=img_size)

# Example usage
fine_tune_yolov10("yolov10n.yaml", "data.yaml", 100, 640)

