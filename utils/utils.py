import torch

from models.yolov3.utils.utils import rescale_boxes


def convert_yolo_outputs_to_coco_standard(yolo_outputs_to_be_converted, current_img_dim, original_img_dimensions,
                                          rescale=None):
    converted_yolo_outputs = []
    for i, output in enumerate(yolo_outputs_to_be_converted):
        converted_yolo_output = dict()
        if output is not None:
            if rescale:
                output = rescale_boxes(output, current_img_dim, original_img_dimensions[i])
            converted_yolo_output["boxes"] = output[:, :4]
            scores = output[:, 4:5]
            scores = torch.transpose(scores, 1, 0)
            converted_yolo_output["scores"] = torch.squeeze(scores, dim=0)
            labels = output[:, -1:].long() + 1     # Coco and Yolo labels are shifted by one
            labels = torch.transpose(labels, 1, 0)
            converted_yolo_output["labels"] = torch.squeeze(labels, dim=0)
        else:
            converted_yolo_output["boxes"] = torch.empty(0, 4)
            converted_yolo_output["labels"] = torch.empty(0)
            converted_yolo_output["scores"] = torch.empty(0)
        converted_yolo_outputs.append(converted_yolo_output)

    return converted_yolo_outputs


