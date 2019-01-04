import os
from multiprocessing import Pool
from glob import glob

from cv2 import cv2


def get_face_images(video_path):
    cascade_path = 'D:\\opencv\\opencv3.1\\build\etc\\haarcascades\\haarcascade_frontalface_alt.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    capture = cv2.VideoCapture(video_path)
    result_list = []
    count = 0
    while(True):
        ret, frame = capture.read()
        count += 1
        if not ret:
            break
            
        if count % 60 != 0:
            continue

        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_region = cascade.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

        if len(face_region) > 0:
            for (x, y, w, h) in face_region:
                result_list.append(frame[y: y + h, x: x + w])

        resized_list = list(map(resize, result_list))
        [save_image(v, i) for i, v in enumerate(resized_list)]

    return True


def save_image(src, index):
    print(f'Save {os.getpid()}')
    cv2.imwrite(f'output\\{os.getpid()}_{index}.png', src)

def resize(src, dst_image_size=64):
    return cv2.resize(src, (dst_image_size, dst_image_size));

def get_video_files(video_path):
    extention = '*mp4'
    result = glob(f'{video_path}\\{extention}')

    return result

def create_directory(path_name):
    if (not os.path.isdir(path_name)):
        os.mkdir(path_name)

def main():
    # TODO: Args + ImageSize
    video_path = 'D:\\project\\dc\\res'
    output_path = 'output'

    create_directory(output_path)

    video_list = get_video_files(video_path)
    print(video_list)

    with Pool(processes=len(video_list)) as pool:
        result = pool.map(get_face_images, video_list)
        result.wait ()


main() if __name__ == '__main__' else None