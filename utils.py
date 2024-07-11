import torch
import torch.nn.functional as F

def denormalize_coordinates(coords, original_width, original_height):
    coords[:, :, 0] = coords[:, :, 0] * original_width
    coords[:, :, 1] = coords[:, :, 1] * original_height
    return coords

def final_displacement_error(gt, output, original_width, original_height):
    gt_denormalized = denormalize_coordinates(gt.clone(), original_width, original_height)
    output_denormalized = denormalize_coordinates(output.clone(), original_width, original_height)
    
    gt_last_boxes = gt_denormalized[-1, 0, :2] 
    output_last_boxes = output_denormalized[-1, 0, :2] 
    
    fde = torch.sqrt(F.mse_loss(gt_last_boxes, output_last_boxes, reduction='none').sum())
    
    return fde.item()

def average_displacement_error(gt, output, original_width, original_height):
    gt_denormalized = denormalize_coordinates(gt.clone(), original_width, original_height)
    output_denormalized = denormalize_coordinates(output.clone(), original_width, original_height)
    
    ade = F.mse_loss(gt_denormalized[:, :, :2], output_denormalized[:, :, :2], reduction='none').sum(dim=2).sqrt().mean()
    
    return ade.item()


def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    y_min = max(box1[0], box2[0])
    x_min = max(box1[1], box2[1])
    y_max = min(box1[0] + box1[2], box2[0] + box2[2])
    x_max = min(box1[1] + box1[3], box2[1] + box2[3])
    
    # Calculate intersection area
    intersection_area = max(0, y_max - y_min) * max(0, x_max - x_min)
    
    # Calculate box areas
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    
    # Calculate union area
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou

def final_intersection_over_union(gt, output):
    # Extract the last bounding box from Ground Truth and Output tensors
    gt_last_box = gt[0][-1].tolist()  # Convert to list for easier indexing
    output_last_box = output[0][-1].tolist()  # Convert to list for easier indexing
    
    iou = calculate_iou(gt_last_box, output_last_box)
    
    return iou

def average_intersection_over_union(gt, output):
    total_iou = 0.0
    num_samples = gt.shape[0]

    for i in range(num_samples):
        sample_gt = gt[i].tolist()
        sample_output = output[i].tolist()

        sample_iou = 0.0
        for j in range(len(sample_gt)):
            sample_iou += calculate_iou(sample_gt[j], sample_output[j])

        total_iou += (sample_iou / len(sample_gt))

    aiou = total_iou / num_samples
    return aiou

## Assuming gt and output are batches of tensors
## gt.shape = (batch_size, sequence_length, 4) and output.shape = (batch_size, sequence_length, 4)

## Compute Final Intersection over Union (FIoU)
# fiou_result = final_intersection_over_union(gt, output)

# print(f"The Final Intersection over Union is: {fiou_result}")

