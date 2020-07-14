from get_coor_angle import init_model, detect_image_2p
import os
import shutil
import time
import torch
import cv2
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/config_2class.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/jetbot_with_angle.name', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/1205_3_last_yolov3-tiny3-1cls.pt', help='weights path')
    parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    #parser.add_argument('--source', type=str, default='/home/thanhtt/Downloads/jetBot', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='/home/thanhtt/Downloads/yolov3_output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    
    with torch.no_grad():
        model, names, colors, device = init_model(opt.weights, cfg = opt.cfg, names=opt.names)
        cap = cv2.VideoCapture(0)
        while(cap.isOpened()):
          ret, frame = cap.read()
          if ret == True:
            #cv2.imshow('Frame',frame)
            #coor, angle, new_img = get_coor_angle(model, frame)
            cv2.imshow('Frame',frame)
            start = time.time()
            a, b, c = detect_image_2p(frame, model, names, device, is_save=False, path_save=None, max_error_ratio_2p=4.0)
            print(a, b, c)
            end = time.time()
            print(end-start)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
          else:
            break
        cap.release()
        cv2.destroyAllWindows()