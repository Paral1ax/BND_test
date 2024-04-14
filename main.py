import cv2 as cv
from ultralytics import YOLO
from ultralytics.models import yolo

from ultralytics.utils.checks import check_imshow
from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict
import numpy as np
from roboflow import Roboflow

model = YOLO("best_yolo8m_medium.pt")
names = model.model.names
video = 'crowd.mp4'

def read_video(path):
    capture = cv.VideoCapture(path)
    assert capture.isOpened(), "Opening video error"

    w, h, fps = (int(capture.get(x)) for x in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FPS))

    result = cv.VideoWriter("people_tracking_yolo8medium.avi",
                            cv.VideoWriter.fourcc(*'mp4v'),
                            fps,
                            (w, h))

    return capture, result


def track_people(capture, video_writer):
    track_history = defaultdict(lambda: [])

    while capture.isOpened():
        success, frame = capture.read()
        if success:
            results = model.track(frame, persist=True, verbose=False)
            boxes = results[0].boxes.xyxy.cpu()

            if results[0].boxes.id is not None:

                # Extract prediction results
                clss = results[0].boxes.cls.cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confs = results[0].boxes.conf.float().cpu().tolist()

                # Annotator Init
                annotator = Annotator(frame, line_width=2)

                for box, cls, track_id in zip(boxes, clss, track_ids):
                    annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

                    # Store tracking history
                    track = track_history[track_id]
                    track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                    if len(track) > 30:
                        track.pop(0)

                    # Plot tracks
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
                    cv.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

            video_writer.write(frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    video_writer.release()
    capture.release()
    cv.destroyAllWindows()

    return track_history

if __name__ == '__main__':
    captured, video_writer = read_video(video)
    track_history = track_people(captured, video_writer)
