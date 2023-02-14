# Copyright 2022 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import argparse
import os
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
from clip_retrieval.clip_client import ClipClient


def retrieve(target_name, outpath, num_class_images):
    num_images = 2*num_class_images
    client = ClipClient(url="https://knn.laion.ai/knn-service", indice_name="laion_400m", num_images=num_images,  aesthetic_weight=0.1)
    
    if len(target_name.split()):
        target = '_'.join(target_name.split())
    else:
        target = target_name
    os.makedirs(f'{outpath}/{target}', exist_ok=True)

    if len(list(Path(f'{outpath}/{target}').iterdir())) >= num_class_images:
        return

    while True:
        results = client.query(text=target_name)
        if len(results) >= num_class_images or num_images > 1e4:
            break
        else:
            num_images = int(1.5*num_images)
            client = ClipClient(url="https://knn.laion.ai/knn-service", indice_name="laion_400m", num_images=num_images,  aesthetic_weight=0.1)

    count = 0
    urls = []
    captions = []

    for each in results:
        name = f'{outpath}/{target}/{count}.jpg'
        success = True
        while True:
            try:
                img = requests.get(each['url'])
                success = True
                break
            except:
                print("Connection refused by the server..")
                success = False
                break
        if success and img.status_code == 200:
            print(len(img.content), count)
            try:
                _ = Image.open(BytesIO(img.content))
                with open(name, 'wb') as f:
                    f.write(img.content)
                urls.append(each['url'])
                captions.append(each['caption'])
                count += 1
            except:
                print("not an image")
        if count > num_class_images:
            break

    with open(f'{outpath}/caption.txt', 'w') as f:
        for each in captions:
            f.write(each.strip() + '\n')

    with open(f'{outpath}/urls.txt', 'w') as f:
        for each in urls:
            f.write(each.strip() + '\n')

    with open(f'{outpath}/images.txt', 'w') as f:
        for p in range(count):
            f.write(f'{outpath}/{target}/{p}.jpg' + '\n')


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--target_name', help='target string for query',
                        type=str)
    parser.add_argument('--outpath', help='path to save retrieved images', default='./',
                        type=str)
    parser.add_argument('--num_class_images', help='number of retrieved images', default=200,
                        type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    retrieve(args.target_name, args.outpath, args.num_class_images)
