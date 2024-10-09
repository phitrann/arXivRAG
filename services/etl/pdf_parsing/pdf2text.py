import argparse
from datetime import datetime
import time
from typing import List, Tuple, Dict
import yaml
import os


from modules.layoutlmv3.model_init import Layoutlmv3_Predictor
import numpy as np
from PIL import Image
import pymupdf
from ultralytics import YOLO

import warnings
warnings.filterwarnings('ignore')

# Define content types
CONTENT_TYPES = [
    "title", "plain_text", "abandon", "figure", "figure_caption", "table", "table_caption", "table_footnote",
    "isolate_formula", "formula_caption", "unknown1", "unknown2", "unknown3", "inline_formula", "isolated_formula", "ocr_text"
]


class ContentBox:
    def __init__(self, rect: pymupdf.Rect, content_type: str):
        self.rect = rect
        self.content_type = content_type

    def __repr__(self):
        return f"ContentBox({self.content_type}, {self.rect})"

def column_boxes(
    page: pymupdf.Page,
    *,
    footer_margin: int = 50,
    header_margin: int = 50,
    no_image_text: bool = True,
    mfd_model: YOLO = None,
    layout_model: Layoutlmv3_Predictor = None,
    mfd_conf_thres: float = 0.25,
    mfd_iou_thres: float = 0.45,
    img_size: int = 640,
) -> List[ContentBox]:
    """Determine content boxes which wrap different types of content on the page using layout detection.

    Args:
        page: PyMuPDF page object
        footer_margin: ignore content if distance from bottom is less
        header_margin: ignore content if distance from top is less
        no_image_text: ignore text inside image bboxes
        mfd_model: pre-initialized YOLO model for math formula detection
        layout_model: pre-initialized Layoutlmv3_Predictor for layout detection
        mfd_conf_thres: confidence threshold for math formula detection
        mfd_iou_thres: IOU threshold for math formula detection
        img_size: image size for YOLO model input
    """
    # Compute relevant page area
    clip = page.rect
    clip.y1 -= footer_margin  # Remove footer area
    clip.y0 += header_margin  # Remove header area

    # Convert page to image
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img_array = np.array(img)

    # Perform layout detection
    layout_res = layout_model(img_array, ignore_catids=[])

    # Perform math formula detection
    mfd_res = mfd_model.predict(img_array, imgsz=img_size, conf=mfd_conf_thres, iou=mfd_iou_thres, verbose=False)[0]

    # Combine layout and math formula detection results
    content_boxes = []
    for item in layout_res['layout_dets']:
        rect = pymupdf.Rect(item['poly'][0], item['poly'][1], item['poly'][4], item['poly'][5])
        content_type = CONTENT_TYPES[item['category_id']]
        content_boxes.append(ContentBox(rect, content_type))

    # Add math formula detection results
    for xyxy, conf, cla in zip(mfd_res.boxes.xyxy.cpu(), mfd_res.boxes.conf.cpu(), mfd_res.boxes.cls.cpu()):
        xmin, ymin, xmax, ymax = [int(p.item()) for p in xyxy]
        rect = pymupdf.Rect(xmin, ymin, xmax, ymax)
        content_type = "inline_formula" if int(cla.item()) == 0 else "isolated_formula"
        content_boxes.append(ContentBox(rect, content_type))

    # Remove content boxes outside the clip area
    content_boxes = [box for box in content_boxes if clip.intersects(box.rect)]

    # Sort content boxes by y0, then x0
    content_boxes.sort(key=lambda box: (box.rect.y0, box.rect.x0))

    # Perform post-processing
    content_boxes = join_content_boxes(content_boxes)
    # content_boxes = handle_overlaps(content_boxes)
    content_boxes = sorted(content_boxes, key=lambda box: (box.rect.y0, box.rect.x0))

    return content_boxes

def join_content_boxes(boxes: List[ContentBox]) -> List[ContentBox]:
    """Join adjacent content boxes of the same type."""
    joined_boxes = []
    current_box = None

    for box in boxes:
        if current_box is None:
            current_box = box
        elif (current_box.content_type == box.content_type and
              abs(current_box.rect.y1 - box.rect.y0) <= 5 and
              abs(current_box.rect.x0 - box.rect.x0) <= 5):
            current_box.rect |= box.rect
        else:
            joined_boxes.append(current_box)
            current_box = box

    if current_box:
        joined_boxes.append(current_box)

    return joined_boxes


def handle_overlaps(boxes: List[ContentBox]) -> List[ContentBox]:
    """Handle overlapping content boxes based on content type priority."""
    priority_order = {
        "title": 1,
        "figure": 2,
        "table": 3,
        "isolate_formula": 4,
        "isolated_formula": 4,  # Same priority as isolate_formula
        "inline_formula": 5,
        "figure_caption": 6,
        "table_caption": 7,
        "formula_caption": 8,
        "plain_text": 9,
        "table_footnote": 10,
        "ocr_text": 11,
        "abandon": 12,
        "unknown1": 13,
        "unknown2": 14,
        "unknown3": 15,
    }

    # Add a default priority for any content types not explicitly listed
    default_priority = max(priority_order.values()) + 1

    resolved_boxes = []
    for box in boxes:
        overlaps = [b for b in resolved_boxes if box.rect.intersects(b.rect)]
        if not overlaps:
            resolved_boxes.append(box)
        else:
            for overlap in overlaps:
                box_priority = priority_order.get(box.content_type, default_priority)
                overlap_priority = priority_order.get(overlap.content_type, default_priority)
                if box_priority < overlap_priority:
                    # Current box has higher priority, adjust or remove the overlapping box
                    if box.rect.contains(overlap.rect):
                        resolved_boxes.remove(overlap)
                    else:
                        # Adjust the overlapping box
                        adjusted_rect = adjust_rect(overlap.rect, box.rect)
                        if adjusted_rect:
                            overlap.rect = adjusted_rect
                        else:
                            resolved_boxes.remove(overlap)
                else:
                    # Adjust the current box
                    adjusted_rect = adjust_rect(box.rect, overlap.rect)
                    if adjusted_rect:
                        box.rect = adjusted_rect
                    else:
                        break
            else:
                resolved_boxes.append(box)

    return sorted(resolved_boxes, key=lambda box: (box.rect.y0, box.rect.x0))

# The rest of your code remains the same

def adjust_rect(rect_to_adjust: pymupdf.Rect, fixed_rect: pymupdf.Rect) -> pymupdf.Rect:
    """Adjust rect_to_adjust to not overlap with fixed_rect."""
    if rect_to_adjust.y0 < fixed_rect.y1 <= rect_to_adjust.y1:
        return pymupdf.Rect(rect_to_adjust.x0, fixed_rect.y1, rect_to_adjust.x1, rect_to_adjust.y1)
    elif rect_to_adjust.y0 <= fixed_rect.y0 < rect_to_adjust.y1:
        return pymupdf.Rect(rect_to_adjust.x0, rect_to_adjust.y0, rect_to_adjust.x1, fixed_rect.y0)
    elif rect_to_adjust.x0 < fixed_rect.x1 <= rect_to_adjust.x1:
        return pymupdf.Rect(fixed_rect.x1, rect_to_adjust.y0, rect_to_adjust.x1, rect_to_adjust.y1)
    elif rect_to_adjust.x0 <= fixed_rect.x0 < rect_to_adjust.x1:
        return pymupdf.Rect(rect_to_adjust.x0, rect_to_adjust.y0, fixed_rect.x0, rect_to_adjust.y1)
    return None


TEXT_CONTENT_TYPES = ["title", "plain_text"]
def visualize_bboxes(input_filename, output_filename, text_filename, footer_margin, header_margin, mfd_model, layout_model):
    doc = pymupdf.open(input_filename) 
    for page_num, page in enumerate(doc):
        bboxes = column_boxes(
            page,
            footer_margin=footer_margin,
            header_margin=header_margin,
            mfd_model=mfd_model,
            layout_model=layout_model
        )
        
        shape = page.new_shape()
        
        for i, bbox in enumerate(bboxes):
            if isinstance(bbox, ContentBox):  # Assuming ContentBox is the correct type
                rect = bbox.rect
                content_type = bbox.content_type
                if content_type in TEXT_CONTENT_TYPES:
                    with open(text_filename, "a") as f:
                        f.write(page.get_text(clip=rect, sort=True))
                        f.write("\n")
            elif isinstance(bbox, pymupdf.Rect):
                rect = bbox
                content_type = "Unknown"
            elif isinstance(bbox, (tuple, list)) and len(bbox) == 4:
                rect = pymupdf.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                content_type = "Unknown"
            else:
                print(f"Skipping invalid bbox at index {i} on page {page_num + 1}: {bbox}")
                continue
            
            shape.draw_rect(rect)
            
            
            # Add both index and content type to the text
            text = f"{i}: {content_type}"
            shape.insert_text(rect.tl + (5, 15), text, fontsize=8, color=(1, 0, 0))  # Red text
        
        shape.finish(width=0.5, color=(1, 0, 0))  # Red line, 0.5 width
        shape.commit()
    
    doc.save(output_filename)
    doc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="output/original_file.pdf", help="Path to the input PDF file")
    parser.add_argument("--footer_margin", type=int, default=50, help="Footer margin")
    parser.add_argument("--header_margin", type=int, default=50, help="Header margin")
    args = parser.parse_args()
    
    # Load model configurations
    with open('configs/model_configs.yaml') as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    
    if os.path.exists(model_configs['model_args']['mfd_weight']) and os.path.exists(model_configs['model_args']['layout_weight']):
        pass 
    else:
        from huggingface_hub import snapshot_download
        # Download the Layout model
        snapshot_download(
            repo_id="opendatalab/PDF-Extract-Kit",
            allow_patterns="models/Layout/*",
            local_dir="models/Layout"
        )

        # Download the MFD model
        snapshot_download(
            repo_id="opendatalab/PDF-Extract-Kit",
            allow_patterns="models/MFD/*",
            local_dir="models/MFD"
        )

    mfd_model = YOLO(model_configs['model_args']['mfd_weight'])
    layout_model = Layoutlmv3_Predictor(model_configs['model_args']['layout_weight'])
    
    # Define output file names
    output_file = args.input_file.replace(".pdf", "-visualized.pdf")
    text_filename = args.input_file.replace(".pdf", "-textbox.txt")
    
    start = time.time()
    
    visualize_bboxes(args.input_file, 
                     output_file, 
                     text_filename, 
                     args.footer_margin, 
                     args.header_margin, 
                     mfd_model, 
                     layout_model)
    
    end = time.time()
    print('Finished! time cost:', int(end-start), 's')
