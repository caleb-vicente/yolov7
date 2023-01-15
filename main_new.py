import torch
import argparse
import numpy as np

import skvideo
skvideo.setFFmpegPath('C:/ffmpeg/bin')
import skvideo.io
import matplotlib.pyplot as plt

from detect_new import detect, draw_bounding_boxes_yolo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=r'C:\Users\caleb\Downloads\58168_003392_Sideline_edited.mp4',
                        help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        det_all_frames = detect(source=opt.source,
                                weights=opt.weights,
                                imgsz=opt.img_size,
                                trace=opt.no_trace,
                                device=opt.device,
                                augment=opt.augment,
                                conf_thres=opt.conf_thres,
                                iou_thres=opt.iou_thres,
                                classes=opt.classes,
                                agnostic_nms=opt.agnostic_nms
                                )

    videodata = skvideo.io.vread(r"C:\Users\caleb\Downloads\58168_003392_Sideline_edited.mp4")
    frame = videodata[304, :, :, :]
    frame = np.transpose(frame, (2, 0, 1))
    frame_tf = torch.from_numpy(frame)
    output = draw_bounding_boxes_yolo()

