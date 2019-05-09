import cv2
import argparse
import time
from datetime import datetime
import numpy as np
from threading import Thread
from threading import Event
from collections import deque
import tensorflow as tf
import math

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class FileVideoStream:
    def __init__(self, path):
        self.stream = cv2.VideoCapture(path)
        self.timeForFrame = 1 / self.stream.get(cv2.CAP_PROP_FPS)
        self.event = Event()

        print(self.timeForFrame)
        self.moreFrames = True
        self.Q = deque(maxlen=300)

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        frameNo = 0
        while True:

            if waitForModelToStart == True and len(self.Q) > 5:
                self.event.wait()


            startTime = time.time()
            (grabbed,frame) = self.stream.read()

            if not grabbed:

                print('no more frames')
                self.moreFrames = False
                return

            frame = frame[500:1100, 900:1500]


            endTime = time.time()
            processingTime = endTime - startTime
            #time.sleep(0.017)
            #print(self.timeForFrame - processingTime)
            if processingTime < self.timeForFrame:
                time.sleep(self.timeForFrame - processingTime)
            self.Q.append(frame)
            frameNo += 1
            #print('frame No.{} appended'.format(frameNo))


    def read(self):
        return self.Q.pop()

    def more(self):
        return self.moreFrames


def detect(img):


    centroids = []
    rows = img.shape[0]
    cols = img.shape[1]
    inp = cv2.resize(img, (600, 600))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB


    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                  sess.graph.get_tensor_by_name('detection_scores:0'),
                  sess.graph.get_tensor_by_name('detection_boxes:0'),
                  sess.graph.get_tensor_by_name('detection_classes:0')],
                 feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]

        if score > 0.3:
            if classId == 1:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                centroid = ((right + x)//2, (bottom + y)//2)
                centroids.append(centroid)
                cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), 2)

                Y = y - 15 if y - 15 > 15 else y + 15
                score = score * 100

                cv2.putText(img, str(round(score, 2)), (int(x), int(Y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 255, 51), 2)
    return(img, centroids)

def distance(p1, p2, type='euclidian', x_weight=1.0, y_weight=1.0):
    if type == 'euclidian':
        return math.sqrt(float((p1[0] - p2[0])**2) / x_weight + float((p1[1] - p2[1])**2) / y_weight)

def tracking(points, paths):
    #path_size = args[""]
    #max_dist = args[""]
    path_size = 7
    max_dist = 50

    #if no previous paths detected, each point starts its own new path
    if not paths:
        for match in points:
            paths.append([match])

    # if there were previous paths
    else:
        #craete array to store newly merged paths
        new_paths = []


        for path in paths:
            _min = 99999
            _match = None

            #this cycle loops though every point and tries to
            #match each point with closest previous path
            #if it is not too far away
            for p in points:

                #if path had only 1 point in it, measure dist to that point
                if len(path) == 1:
                    dist = distance(p, path[-1])

                #if path had 2 or more points, measure dist to point "expected" to be
                else:
                    xn = 2 * path[-1][0] - path[-2][0]
                    yn = 2 * path[-1][1] - path[-2][1]
                    dist = distance(p, (xn, yn))

                #this just finds the smallest distance between each point and path to match them
                if dist < _min:
                    _min = dist
                    _match = p

            # if point was matched and it wasnt too far away update path with its new added point
            if _match and _min <= max_dist:
                points.remove(_match)
                path.append(_match)
                new_paths.append(path)

            # do not drop path if current frame has no matches
            # might be just flicker in detection
            if _match is None:
                new_paths.append(path)

        #paths are now updated
        paths = new_paths

        # unmatched pathless points create new paths
        if len(points):
            for p in points:

                #need if statement code for exit zone there

                paths.append([p])

    # save only last N points in path_size
    for i, _ in enumerate(paths):
        paths[i] = paths[i][path_size * -1:]

    return paths

def visualize_paths(frame, paths):

    for path in paths:

        for i in range(1,len(path)):
            cv2.line(frame, (int(path[i][0]), int(path[i][1])), (int(path[i-1][0]), int(path[i-1][1])), (0,255,0), 2)

    return frame

def counting(frame, paths, polygon, line):

    polyTrigger = False
    lineTrigger = False
    global came
    global left



    for path in paths:
        path = path[-5:]

        #d = (path[-1][0] - line[0][0]) * (line[1][1] - line[0][1]) - (path[-1][1] - line[0][1]) * (line[1][0] - line[0][0])

        sides = []
        for point in path:
            sides.append((point[0] - line[0][0]) * (line[1][1] - line[0][1]) - (point[1] - line[0][1]) * (line[1][0] - line[0][0]))

        if (
        len(sides) == 5 and
        sides[4] > 0 and
        sides[3] <= 0 and
        sides[2] < 0 and
        sides[1] < 0 and
        sides[0] < 0 and
        polygon.contains(Point(path[-1][0], path[-1][1]))
        ):
            left += 1
            lineTrigger = True
        if (
        len(sides) == 5 and
        sides[4] < 0 and
        sides[3] >= 0 and
        sides[2] > 0 and
        sides[1] > 0 and
        sides[0] > 0 and
        polygon.contains(Point(path[-1][0], path[-1][1]))
        ):
            came += 1
            lineTrigger = True





        if polygon.contains(Point(path[-1][0], path[-1][1])):
            polyTrigger = True




    overlayRed = frame.copy()
    overlayGreen = frame.copy()
    alpha=0.2
    cv2.fillPoly(overlayRed, vertices, (0,0,255))
    cv2.fillPoly(overlayGreen, vertices, (0,255,0))

    if lineTrigger:
        cv2.line(frame, line[0], line[1], (255, 0 ,0), 2)
        lineTrigger = False
    else:
        cv2.line(frame, line[0], line[1], (0, 255 ,0), 2)

    if polyTrigger:
        cv2.addWeighted(overlayGreen, alpha, frame, 1 - alpha, 0, frame)
    else:
        cv2.addWeighted(overlayRed, alpha, frame, 1 - alpha, 0, frame)
    #print(came)
    cameText = 'people came: ' + str(came)
    leftText = 'people left: ' + str(left)

    cv2.putText(frame, cameText, (20, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, leftText, (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

    return frame




ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=int, default= 0,
    help="number of model to use - 0: MobileNetSSD, 1: PELEE")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
ap.add_argument("-v", "--video", help="path to video file")
args = vars(ap.parse_args())

if args["model"] == 0:
    MODEL_NAME = 'models/mobileNetSSD/frozen_inference_graph.pb'
    print('running with MobileNet SSD')
if args["model"] == 1:
    MODEL_NAME = 'models/faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28/frozen_inference_graph.pb'
    print('running with faster RCNN inception resnet low proposals model')
if args["model"] == 2:
    MODEL_NAME = 'models/faster_rcnn_resnet101_lowproposals_coco_2018_01_28/frozen_inference_graph.pb'
    print('running with faster rcnn resnet101 low proposals model')
if args["model"] == 3:
    MODEL_NAME = 'models/rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb'
    print('running with r-fcn model')
if args["model"] == 4:
    MODEL_NAME = 'models/first_trained_model/frozen_inference_graph.pb'
    print('running with first trained model')
if args["model"] == 5:
    MODEL_NAME = 'models/6841_model/frozen_inference_graph.pb'
    print('running with 6841_trained model')
if args["model"] == 6:
    MODEL_NAME = 'models/big_one2_model/frozen_inference_graph.pb'
    print('running with big_one2 model')
if args["model"] == 7:
    MODEL_NAME = 'models/paintedSSDmodel/frozen_inference_graph.pb'
    print('running with paintedSSD model')
if args["model"] == 8:
    MODEL_NAME = 'models/faster_rcnn_trained_model/frozen_inference_graph.pb'
    print('running with faster_rcnn_trained_model ')    
    
    

with tf.gfile.FastGFile(MODEL_NAME, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())




came = 0
left = 0

waitForModelToStart = True
fvs = FileVideoStream(args["video"]).start()

outputpath = 'output/out.mp4'
#out = cv2.VideoWriter('output/{}_{}.mp4'.format(args["model"],args["video"]),cv2.VideoWriter_fourcc(*'mp4v'), 10, (600,580))
out = cv2.VideoWriter(outputpath,cv2.VideoWriter_fourcc(*'mp4v'), 10, (600,580))

time.sleep(1.0)
pulledFrame = 0
paths = []

savedFrames = 0

vertices = np.array([
    [[360, 220], [550, 300], [400, 570], [200, 470]]
    ])
polygon = Polygon([(360, 220),(550, 300), (400, 570),(200, 470)])

line = [(580,0), (260, 600)]

#polygon = Polygon(vertices)



with tf.Session() as sess:
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')





    while fvs.more():


        frame = fvs.read()

        pulledFrame += 1
        #print('Pulled frame No. {}'.format(pulledFrame))
        
        fps_start = time.time()
        
        frame, centroids = detect(frame)

        waitForModelToStart = False
        fvs.event.set()


        paths = tracking(centroids, paths)
        frame = visualize_paths(frame, paths)
        frame = counting(frame, paths, polygon, line)
        
  
        fps_end = time.time()
        time_delta = fps_end - fps_start
        cv2.putText(frame, 'fps {}'.format(round(1/time_delta)) , (20, 90), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)  
        
        
        out.write(frame)

        #cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


print('came {}'.format(came))
print('left {}'.format(left))
with open('test_results.txt','a') as f:
        f.write('\n')
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n')
        f.write(MODEL_NAME + '\n')
        f.write(args["video"] + '\n')
        f.write('came {}'.format(came) + '\n')
        f.write('left {}'.format(left) + '\n')
        f.write('------------------------')
        f.close()

out.release()
cv2.destroyAllWindows
