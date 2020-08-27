import argparse
import requests
import cv2
import dlib
import os
import re
import shutil
import sys
from glob import glob
from googleapiclient.discovery import build

# API Key
GOOGLE_API_KEY = '_key_'
CUSTOM_SEARCH_ENGINE_ID = '_id_'
# Model定義
FACE_CASCADE_XML = './xml/haarcascade_frontalface_default.xml'
CNN_FACE_DETECTOR_DAT = './dat/mmod_human_face_detector.dat'
# デフォルトディレクトリ
DEFAULT_INPUT_DIR = './input'
DEFAULT_OUTPUT_DIR = './output'
TEMP_DIR = './.tmp'
# 信頼度の閾値
CONFIDENCE = 0.5

def parse_args():
    # オプション定義
    usage = 'python {} [-n|--number NUMBER] [-f|--file FILE] [--input INPUT] [--output OUTPUT] keyword'.format(__file__)
    argparser = argparse.ArgumentParser(usage=usage)
    argparser.add_argument('keyword', nargs='?', help='search keyword')
    argparser.add_argument('-n', '--number', type=int, default=10, help='number of search images')
    argparser.add_argument('-f', '--file', type=str, help='the path of a file which contains urls')
    argparser.add_argument('--input', type=str, default=DEFAULT_INPUT_DIR,
        help="file input directory. if not specified, the default path is '{}'".format(DEFAULT_INPUT_DIR))
    argparser.add_argument('--output', type=str, default=DEFAULT_OUTPUT_DIR,
        help="file output directory. if not specified, the default path is '{}'".format(DEFAULT_OUTPUT_DIR))
    # オプション解析
    return argparser.parse_args()

# モザイク処理(OpenCV)
def mosaic_img(img_path, dest):
    src = cv2.imread(img_path)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_XML)
    faces = face_cascade.detectMultiScale(src_gray)
    if len(faces) > 0:
        for x, y, w, h in faces:
            print(x, y, w, h)
            src = mosaic_area(src, x, y, w, h)
    cv2.imwrite(dest, src)

# モザイク処理(Dlib)
def mosaic_img_with_cnn(img_path, dest):
    face_detector = dlib.cnn_face_detection_model_v1(CNN_FACE_DETECTOR_DAT)

    img = dlib.load_rgb_image(img_path)
    dets = face_detector(img, 1)

    src = cv2.imread(img_path)
    if len(dets) > 0:
        for d in dets:
            if d.confidence < CONFIDENCE:
                continue
            x1 = d.rect.left()
            x2 = d.rect.right()
            y1 = d.rect.top()
            y2 = d.rect.bottom()
            src =  mosaic_area(src, x1, y1, y2 - y1, x2 -x1)
    cv2.imwrite(dest, src)

def mosaic_area(src, x, y, width, height, ratio=0.08):
    dst = src.copy()
    dst[y:y + height, x:x + width] = mosaic(dst[y:y + height, x:x + width], ratio)
    return dst

def mosaic(src, ratio):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

# ファイルURL読み込み
def read_and_write_from_file_urls(file_path, output_dir):
    urls = read_urls_from_file(file_path)
    download_and_write_from_urls(urls, output_dir)

def read_urls_from_file(file_path):
    try:
        with open(file_path, 'r') as f:
            urls = f.readlines()
    except IOError:
        sys.exit("{} doesn't exist or can't be opend".format(file_path))
    return [x.strip() for x in urls]

def download_and_write_from_urls(urls, output_dir):
    if not os.path.exists(TEMP_DIR):
        # 一時ディレクトリの作成
        os.mkdir(TEMP_DIR)
    for url in urls:
        try:
            response = requests.get(url)
            image = response.content
            file_name = url.split('/')[-1]
            write_path = '{}/{}'.format(TEMP_DIR, file_name)
            with open(write_path, 'wb') as f:
                f.write(image)
            # mosaic_img(write_path, '{}/{}'.format(output_dir, file_name))
            mosaic_img_with_cnn(write_path, '{}/{}'.format(output_dir, file_name))
        except requests.exceptions.RequestException as e:
            print("エラー : ", e, file=sys.stderr)
    
    # 一時ディレクトリ削除
    shutil.rmtree(TEMP_DIR)


# Google画像読み込み
def read_and_write_from_keyword(keyword, number, output_dir):
    urls = get_img_urls_from_google(keyword, number)
    download_and_write_from_urls(urls, output_dir)

def get_img_urls_from_google(keyword, number):
    service = build('customsearch', 'v1', developerKey=GOOGLE_API_KEY)
    page_limit = number / 10 if number % 10 == 0 else number // 10 + 1
    startIndex = 1
    responses = []
    for page in range(0, int(page_limit)):
        try:
            res = service.cse().list(
                q=keyword,
                cx=CUSTOM_SEARCH_ENGINE_ID,
                lr='lang_ja',
                num=10,
                start=startIndex,
                searchType='image'
            ).execute()
            if not 'items' in res:
                break
            responses.append(res)
            startIndex = res.get('queries').get('nextPage')[0].get('startIndex')
        except Exception as e:
            sys.exit(e)

    urls = []
    for res in responses:
        for item in res['items']:
            urls.append(item['link'])
    return urls[:number]

# ディレクトリ読み込み
def read_and_write_from_dir(input_dir, output_dir):
    files = get_file_list(input_dir)
    for file in files:
        output_file = '{}/{}'.format(output_dir, os.path.basename(file))
        # mosaic_img(file, output_file)
        mosaic_img_with_cnn(file, output_file)

def get_file_list(input_dir):
    files = []
    for path in glob('{}/*'.format(input_dir)):
        if os.path.isfile(path):
            files.append(path)
    return files

def main():
    if not os.path.exists(DEFAULT_INPUT_DIR):
        os.mkdir(DEFAULT_INPUT_DIR)
    if not os.path.exists(DEFAULT_OUTPUT_DIR):
        os.mkdir(DEFAULT_OUTPUT_DIR)

    args = parse_args()
    if args.file:
        read_and_write_from_file_urls(args.file, args.output)
    elif args.keyword:
        read_and_write_from_keyword(args.keyword, args.number, args.output)
    else:
        read_and_write_from_dir(args.input, args.output)

main()
