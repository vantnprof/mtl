import argparse
from ultralytics import YOLO
import wandb
import os
import torch


def main(args):

    print(f"Training on: {args.data}")
    print(f"Epochs: {args.epochs}, IoU: {args.iou}, Conf: {args.conf}, NMS: {args.nms}")
    print(f"Using augmentation: {args.aug}")

    wandb.init(
        project=args.project,
        name=args.exp_name,
        config=vars(args)
    )

    # Load a YOLOv11 model (ensure the YAML is correctly set for this model)
    model = YOLO(args.model_config, task="detect")
    resume = False    
    if args.weights:
        pretrained_weights = torch.load(args.weights)['model'].state_dict()
        model.model.load_state_dict(pretrained_weights, strict=False)
        resume = True

    _ = model.train(
        resume=resume,  # Set to True to resume training from a checkpoint
        data=args.data, 
        epochs=args.epochs, 
        imgsz=args.imgsz, 
        batch=args.batch, 
        save_dir=os.path.join(args.save_dir, args.project, args.exp_name),
        project=args.project,
        name=args.exp_name,
        device=args.device,
        iou=args.iou,
        conf=args.conf,
        nms=args.nms,
        augment=args.aug,
        multi_scale=args.multi_scale,
        amp=False,
        lr0=args.lr,  # Initial learning rate
        close_mosaic=0,
        # lambda_orignal_loss=args.lambda_orignal_loss,
        # save=True,
        # val=True,
        # rect=False
    )

    # Export to ONNX
    if args.export:
        model.export(format="onnx")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv11 model with Ultralytics")
    parser.add_argument("--model_config", type=str, default="yolo11x.pt", help="YOLO model config YAML file")
    parser.add_argument("--data", type=str, default=None, help="Dataset YAML file")
    parser.add_argument("--weights", type=str, default=None, help="Path to pretrained weights")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    # parser.add_argument("--lambda_orignal_loss", type=float, default=1.0, help="Weight for original loss")
    parser.add_argument("--infer_img", type=str, default="https://ultralytics.com/images/bus.jpg", help="Image path or URL for inference")
    parser.add_argument("--export", action="store_true", help="Whether to export the model to ONNX")
    parser.add_argument("--project", default="MultilinearTransformationLayer", help="Project name to log to Wandb")
    parser.add_argument("--exp_name", default="YOLOv11_fold0", help="The name of run, include fold index as needed")
    parser.add_argument("--save_dir", default="output/", help="The root of output directory")
    parser.add_argument("--device", type=str, default=torch.device("cuda") if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--iou", type=float, default=0.2, help="IoU threshold")
    parser.add_argument("--conf", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--nms", action="store_true", help="Enable NMS postprocessing")
    parser.add_argument("--aug", action="store_true", help="Enable image augmentations like jumble-up, flips, rotations")
    parser.add_argument("--multi_scale", action="store_true", help="Enable multiscale.")
    args = parser.parse_args()
    main(args)