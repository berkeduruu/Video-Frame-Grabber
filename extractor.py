import cv2

def extract_specific_frame(video_path, ts):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (False, f"Error: Video could not be opened{video_path}", None, -1)

    # Milisaniyeye git
    cap.set(cv2.CAP_PROP_POS_MSEC, ts)
    
    # O anki frame'in numarasını al (tekrarları bulmak için)
    # Not: set'ten sonra bir okuma yapmadan frame numarasını almak
    # bazen tutarsız olabilir, bu yüzden önce okuyup sonra almak daha garanti.
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return (False, f"Error: ms value could not be read", None, -1)

    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    cap.release()
    return (True, "Success", frame, frame_num)