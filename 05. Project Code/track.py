import argparse
import cv2
import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

''' 멀티 프로세스 & EMAIL 보내기 & DB에 데이터 보내기위한 시간 라이브러리 ''' 
from multiprocessing import Process
from collections import deque

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'version') not in sys.path:
    sys.path.append(str(ROOT / 'version'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strongsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strongsort'))  # add strong_sort ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from version.ultralytics.nn.autobackend import AutoBackend
from version.ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams
from version.ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from version.ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from version.ultralytics.yolo.utils.checks import check_file, check_imgsz, check_imshow, print_args, check_requirements
from version.ultralytics.yolo.utils.files import increment_path
from version.ultralytics.yolo.utils.torch_utils import select_device
from version.ultralytics.yolo.utils.ops import Profile, non_max_suppression, scale_boxes, process_mask, process_mask_native
from version.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

from trackers.multi_tracker_zoo import create_tracker
from email_db import *

@torch.no_grad()
def run(
        source='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        tracking_config=None,
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=True,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_trajectories=False,  # save trajectories for each track
        save_vid=True,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs' / 'track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
        
        send_DB = False, # 데이터 베이스로 데이터를 보낼지 말지 
        send_DB_term = 60, # DB로 보내는 시간(초) 단위
        send_EMAIL = False,
        send_EMAIL_term = 3,
        target_object = 'military drone'
):
     
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = yolo_weights.stem
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = Path(yolo_weights[0]).stem
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name else exp_name + "_" + reid_weights.stem
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    is_seg = '-seg' in str(yolo_weights)
    model = AutoBackend(yolo_weights, device=device, dnn=dnn, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_imgsz(imgsz, stride=stride)  # check image size

    '''DB 관련 '''
    is_data_sent = False 
    
    '''EMAIL 관련'''
    is_email_first = True
    address = get_customer()
   
    '''바로 아래 for-loop부터 각 프레임(0.3초?)별로 데이터를 저장 & 업데이터 하기 위한 딕셔너리 타입 '''
    frame_List=dict()
    target_List=dict()
    
    '''TARGET_OBJECT  '''
    t_before = '', ''
    
    # Dataloader
    bs = 1
    if webcam:
        show_vid = check_imshow(warn=True)
        dataset = LoadStreams(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
        bs = len(dataset)
    else:
        dataset = LoadImages(
            source,
            imgsz=imgsz,
            stride=stride,
            auto=pt,
            transforms=getattr(model.model, 'transforms', None),
            vid_stride=vid_stride
        )
    vid_path, vid_writer, txt_path = [None] * bs, [None] * bs, [None] * bs
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup

    # Create as many strong sort instances as there are video sources
    tracker_list = []
    for i in range(bs):
        tracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)
        tracker_list.append(tracker, )
        if hasattr(tracker_list[i], 'model'):
            if hasattr(tracker_list[i].model, 'warmup'):
                tracker_list[i].model.warmup()
    outputs = [None] * bs

    # Run tracking
    #model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile(), Profile())
    curr_frames, prev_frames = [None] * bs, [None] * bs
    
    

    
    for frame_idx, batch in enumerate(dataset):
        
        path, im, im0s, vid_cap, s = batch
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            preds = model(im, augment=augment, visualize=visualize)

        # Apply NMS
        with dt[2]:
            if is_seg:
                masks = []
                p = non_max_suppression(preds[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
                proto = preds[1][-1]
            else:
                p = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        # Process detections
        for i, det in enumerate(p):  # detections per image
            
            '''
            이 for-loop은 한 프레임 안의 각각 개체를 관리하는데 
            아래 리스트에서는 [4, 'military drone', '2023-03-09 18:36:11.50'] 
            <- 이런식으로 트래킹번호, 라벨이름, 디텍팅된 시간을 저장
            ''' 
            
            individual_List=[]
            
            seen += 1
            if webcam:  # bs >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if hasattr(tracker_list[i], 'tracker') and hasattr(tracker_list[i].tracker, 'camera_update'):
                if prev_frames[i] is not None and curr_frames[i] is not None:  # camera motion compensation
                    tracker_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                if is_seg:
                    shape = im0.shape
                    # scale bbox first the crop masks
                    if retina_masks:
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                        masks.append(process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2]))  # HWC
                    else:
                        masks.append(process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True))  # HWC
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], shape).round()  # rescale boxes to im0 size
                else:
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                with dt[3]:
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)
                
                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    
                    if is_seg:
                        # Mask plotting
                        annotator.masks(
                            masks[i],
                            colors=[colors(x, True) for x in det[:, 5]],
                            im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                            255 if retina_masks else im[i]
                        )
                    
                    for j, (output) in enumerate(outputs[i]):
                        # partial_List=[]
                        bbox = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]
                        
                        '''
                        Step 1. 각각 라벨의 정보(라벨명, detecting된 시간)을 individual_list에 넣어서 보냄
                        
                        ex) 현재 이런 식으로 저장됨
                        [[4, 'bird', '2023-03-09 18:36:11.50'], [3, 'military drone', '2023-03-09 18:36:11.50'], [2, 'military drone', '2023-03-09 18:36:11.50'], [1, 'military drone', '2023-03-09 18:36:11.50']]
                        '''
                        
                        tracking_index = int(id)                 
                        tracking_object = names[int(cls)] 
                        
                        # send_database에서 현재 시간 정보를 얻어옴
                        current_time = get_current_time()
                        individual_List.append([tracking_index, tracking_object, current_time])
                        
                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox/seg to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            color = colors(c, True)
                            annotator.box_label(bbox, label, color=color)
                            
                            if save_trajectories and tracking_method == 'strongsort':
                                q = output[7]
                                tracker_list[i].trajectory(im0, q, color=color)
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(np.array(bbox, dtype=np.int16), imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)
                
                '''
                이제 각 라벨별 데이터를 frame_List에 최종적으로 입력 및 
                individaul_List에 저장된 데이터를 clear함
                '''
                
                for numb_track, name_track, tracked_time in individual_List:
                    if numb_track not in list(frame_List.keys()):
                        frame_List[numb_track] = [name_track, tracked_time, tracked_time]  
                        if target_object == name_track:          
                            target_List[numb_track] = [name_track, tracked_time, tracked_time]
                       
                    
                    else:
                       frame_List[numb_track][2] = tracked_time
                       if numb_track in target_List:
                           target_List[numb_track][2] = tracked_time
                           print('\n------numb_track------- : ',numb_track)
                    #elif frame_List
                
                individual_List.clear()
                
            else:
                pass
                #tracker_list[i].tracker.pred_n_update_all_tracks()
  
                
            print('\n-------TARGET-------')
            print(target_List,'\n')
            
            print('\n-------FRAME-------')
            print(frame_List,'\n')
            
            
           
            # Stream results
            im0 = annotator.result()
            if show_vid:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)
            
            '''
            데이터는 여기부터 DB로!! 
            
            frameList = {1: ['helicopter', '2023-03-09 22:44:42.52', '2023-03-09 22:44:43.84']} <- 요런식으로 데이터가 현재 저장되어 있음
            
            step1. 일단 send_DB_term 을 통해서 몇초(DB_sec)에 한번씩 DB로 데이터를 보낼 것인지 세팅하고
            step2. DB로 보낸다면 IF 문을 통해 딱 1번씩만 보내질 수 있도록 설정
            step3. 멀티프로세스를 사용하여 DB_process를 통해 데이터를 DB로 전송
            step4. 사용했던 fram_List 를 update_List 함수를 사용해서 업데이트 해줌
            '''
            
            if send_DB:
                DB_start = 5     
                now_DB = datetime.now() 
                if int(now_DB.second) % send_DB_term == DB_start and is_data_sent == False:
                    
                    # 이때 DB로 데이터 보내기! 
                    frame_List = remove_error(frame_List, current_time)
                    th2 = Process(target = DB_process(frame_List))
                    th2.start()
                    
                    
                    is_data_sent = True
                    print("-------------------------------------------")
                    print("\nData successfully sent to DB! \n")
                    th2.terminate()
                    
                    
                    frame_List = update_List(frame_List, current_time)
                    
                    print("\nData Updated!!\n")
                    print("-------------------------------------------")
                    
                    
                elif int(now_DB.second) % send_DB_term == DB_start and is_data_sent == True:
                    pass 
                elif int(now_DB.second) % send_DB_term != DB_start and is_data_sent == True:
                    is_data_sent = False
                else: 
                    is_data_sent = False
            
            '''
            여기부터는 E-MAIL로!! 

            위의 코드와 매우 비슷함
            step1. 일단 send_EMAIL_term 을 통해서 몇분에 한번씩 E-MAIL를 보낼 것인지 세팅하고
            step2. EMAIL을 보낸다면 IF 문을 통해 ?분당 1번씩만 보내질 수 있도록 설정
            step3. 멀티프로세스를 사용하여 E-MAIL을 발송
            
            target_list =  {2: ['airplane', '2023-03-13 17:06:33.31', '2023-03-13 17:06:43.78']}
            '''
            ######################################
            #          여기부터입니다             #
            ######################################
            
            if send_EMAIL and is_email_first == True: 
                
                if len(target_List)>=1:
                    t_before = current_time 
                    th3 = Process(target = send_alarm(target_object,is_email_first,len(target_List), t_before, current_time, address))
                    th3.start()
                    is_email_first = False 
                    th3.terminate()
                else:
                    pass 
                

            if send_EMAIL and is_email_first == False: 
                
                if len(target_List) >= 1:
                    
                    t_cal = calculate_time(t_before, current_time)
                    
                    if t_cal >= send_EMAIL_term:
                        target_List = remove_error(target_List, current_time)
                        th4 = Process(target = send_alarm(target_object,is_email_first,len(target_List), t_before, current_time, address))
                        th4.start()
                        target_List= update_List(target_List, current_time)
                        t_before = current_time
                        
                        print('\n----------------------------------')
                        print("Email Sented & List Updated!!!!!!\n\n")
                        print(target_List)
                        th4.terminate()   
                    else:
                        pass 
                        
                
                else:
                    pass 
            else: pass
        
            
            prev_frames[i] = curr_frames[i]
            

                
            
        # Print total time (preprocessing + inference + NMS + tracking)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")
    
   
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {tracking_method} update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list((save_dir / 'tracks').glob('*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)
        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / './best.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='ocsort', help='strongsort, ocsort, bytetrack, botsort')
    parser.add_argument('--tracking-config', type=Path, default=None)
    parser.add_argument('--source', type=str, default='./video/test_email.mp4', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true',default=True, help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-trajectories', action='store_true', help='save trajectories for each track')
    parser.add_argument('--save-vid', action='store_true',default=False, help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    parser.add_argument('--send_DB', action='store_true', default=False, help=' T/F send date to DB')
    parser.add_argument('--send_DB_term', action='store_true', default=15, help='term to send data to DB')
    parser.add_argument('--send_EMAIL', action='store_true', default=True, help='T /F send date to EMAIL')
    parser.add_argument('--send_EMAIL_term', action='store_true', default=1, help='term to send EMAIL')
    parser.add_argument('--target_object', action='store_true', default='airplane', help='Target object to count')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    opt.tracking_config = ROOT / 'trackers' / opt.tracking_method / 'configs' / (opt.tracking_method + '.yaml')
    print_args(vars(opt))
    return opt

def track_main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    th1 = Process(target = track_main(opt))
    th1.start()
    
