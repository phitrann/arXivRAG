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

    # with open('content_boxes.txt', 'w') as f:
    #     f.write(str(content_boxes))
    
    # Sort content boxes by y0, then x0
    # content_boxes.sort(key=lambda box: (box.rect.y0, box.rect.x0))

    # Perform post-processing
    content_boxes = join_content_boxes(content_boxes)
    # content_boxes = handle_overlaps(content_boxes)
    content_boxes = sort_boxes(content_boxes, page)
    
    # content_boxes = [
    #     ContentBox(pymupdf.Rect(53.70718765258789, 56.563297271728516, 541.8118896484375, 73.83734130859375), CONTENT_TYPES[0]),
    #     ContentBox(pymupdf.Rect(57.62197494506836, 94.13801574707031, 535.779296875, 166.1094512939453), CONTENT_TYPES[1]),
    #     ContentBox(pymupdf.Rect(146.0485382080078, 193.51214599609375, 190.6062469482422, 205.16891479492188), CONTENT_TYPES[0]),
    #     ContentBox(pymupdf.Rect(49.1270637512207, 218.56829833984375, 287.4342041015625, 289.2501525878906), CONTENT_TYPES[1]),
    #     ContentBox(pymupdf.Rect(48.93668746948242, 289.9808044433594, 287.59381103515625, 480.6813049316406), CONTENT_TYPES[1]),
    #     ContentBox(pymupdf.Rect(49.43186569213867, 481.39520263671875, 287.3237609863281, 587.3452758789062), CONTENT_TYPES[1]),
    #     ContentBox(pymupdf.Rect(50.04110336303711, 596.0108032226562, 127.0931396484375, 608.215087890625), CONTENT_TYPES[0]),
    #     ContentBox(pymupdf.Rect(49.17326354980469, 616.7603149414062, 287.2789001464844, 688.440185546875), CONTENT_TYPES[1]),
    #     ContentBox(pymupdf.Rect(308.791748046875, 194.527587890625, 545.7149047851562, 217.08059692382812), CONTENT_TYPES[1]),
    #     ContentBox(pymupdf.Rect(307.817138671875, 220.33657836914062, 546.3914184570312, 398.9355773925781), CONTENT_TYPES[1]),
    #     ContentBox(pymupdf.Rect(308.36138916015625, 400.6340637207031, 546.3931884765625, 555.3556518554688), CONTENT_TYPES[1]),
    #     ContentBox(pymupdf.Rect(308.356689453125, 556.9380493164062, 545.85107421875, 688.3251342773438), CONTENT_TYPES[1]),
    #     ContentBox(pymupdf.Rect(308.46435546875, 690.7161865234375, 545.8331298828125, 713.3115844726562), CONTENT_TYPES[1]),
    #     ContentBox(pymupdf.Rect(15.702548027038574, 211.5231475830078, 35.39647674560547, 557.5840454101562), CONTENT_TYPES[2]),
    #     ContentBox(pymupdf.Rect(48.9299430847168, 694.2745361328125, 286.93951416015625, 713.212646484375), CONTENT_TYPES[2])
    # ]

    return [box.rect for box in content_boxes if box.content_type in ["plain_text", "title"]]

def sort_boxes(boxes: List[ContentBox], page: pymupdf.Page) -> List[ContentBox]:
    """Sort content boxes for complex layouts, including mixed single and multi-column pages."""
    
    # Separate 'abandon' boxes from others
    abandon_boxes = [box for box in boxes if box.content_type == 'abandon']
    other_boxes = [box for box in boxes if box.content_type != 'abandon']

    # Determine page width
    page_width = page.mediabox_size[0]

    # Determine left and right column boundaries
    # Let's define a column threshold at half of the page width
    column_threshold = page_width * 0.45

    # Function to determine the 'column' of a box
    def get_column(box: ContentBox):
        if box.rect.x0 < column_threshold:
            return 'left'
        else:
            return 'right'

    # Assign column to each box
    for box in other_boxes:
        box.column = get_column(box)  # Dynamically adding an attribute

    # Now group boxes by columns
    left_boxes = [box for box in other_boxes if box.column == 'left']
    right_boxes = [box for box in other_boxes if box.column == 'right']

    # Sort boxes within each group by y0 (top to bottom)
    left_boxes.sort(key=lambda box: box.rect.y0)
    right_boxes.sort(key=lambda box: box.rect.y0)

    # Combine the boxes in reading order
    # First the full-width boxes
    sorted_boxes = left_boxes + right_boxes + abandon_boxes

    # Clean up the dynamically added 'column' attribute
    for box in other_boxes:
        del box.column

    return sorted_boxes

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
def visualize_bboxes(input_filename, output_filename, text_filename, footer_margin, header_margin, img_size, mfd_conf_thres, mfd_iou_thres, mfd_model, layout_model):
    doc = pymupdf.open(input_filename) 
    for page_num, page in enumerate(doc):
        bboxes = column_boxes(
            page,
            footer_margin=footer_margin,
            header_margin=header_margin,
            img_size=img_size,
            mfd_conf_thres=mfd_conf_thres,
            mfd_iou_thres=mfd_iou_thres,
            mfd_model=mfd_model,
            layout_model=layout_model
        )
        
        shape = page.new_shape()
        mid_x = page.mediabox_size[0] * 0.45
        shape.draw_rect(pymupdf.Rect(mid_x, 0,mid_x, page.mediabox_size[1]))
        
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
    
    

with open('configs/model_configs.yaml') as f:
    model_configs = yaml.load(f, Loader=yaml.FullLoader)
    
img_size = model_configs['model_args']['img_size']
mfd_conf_thres = model_configs['model_args']['conf_thres']
mfd_iou_thres = model_configs['model_args']['iou_thres']
device = model_configs['model_args']['device']
dpi = model_configs['model_args']['pdf_dpi']

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



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input_file", type=str, default="output/2312.03441v1.pdf", help="Path to the input PDF file")
#     parser.add_argument("--footer_margin", type=int, default=50, help="Footer margin")
#     parser.add_argument("--header_margin", type=int, default=50, help="Header margin")
#     args = parser.parse_args()
    
#     # Load model configurations
#     with open('configs/model_configs.yaml') as f:
#         model_configs = yaml.load(f, Loader=yaml.FullLoader)
        
#     img_size = model_configs['model_args']['img_size']
#     conf_thres = model_configs['model_args']['conf_thres']
#     iou_thres = model_configs['model_args']['iou_thres']
#     device = model_configs['model_args']['device']
#     dpi = model_configs['model_args']['pdf_dpi']
    
#     if os.path.exists(model_configs['model_args']['mfd_weight']) and os.path.exists(model_configs['model_args']['layout_weight']):
#         pass 
#     else:
#         from huggingface_hub import snapshot_download
#         # Download the Layout model
#         snapshot_download(
#             repo_id="opendatalab/PDF-Extract-Kit",
#             allow_patterns="models/Layout/*",
#             local_dir="models/Layout"
#         )

#         # Download the MFD model
#         snapshot_download(
#             repo_id="opendatalab/PDF-Extract-Kit",
#             allow_patterns="models/MFD/*",
#             local_dir="models/MFD"
#         )

#     mfd_model = YOLO(model_configs['model_args']['mfd_weight'])
#     layout_model = Layoutlmv3_Predictor(model_configs['model_args']['layout_weight'])
    
#     # Define output file names
#     output_file = args.input_file.replace(".pdf", "-visualized.pdf")
#     text_filename = args.input_file.replace(".pdf", "-textbox.txt")
    
#     start = time.time()
#     visualize_bboxes(args.input_file, 
#                         output_file, 
#                         text_filename, 
#                         args.footer_margin, 
#                         args.header_margin, 
#                         img_size,
#                         conf_thres,
#                         iou_thres,
#                         mfd_model, 
#                         layout_model)
#     end = time.time()
#     print('Finished! time cost:', int(end-start), 's')
