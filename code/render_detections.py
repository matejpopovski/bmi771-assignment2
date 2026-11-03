"""Render selected saved predictions with legible labels, without rerunning a model."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import matplotlib
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor
from torchvision.utils import draw_bounding_boxes, save_image
from det_libs.utils import gen_distinct_colors
from det_libs.config import load_config

ROOT=Path(__file__).resolve().parent.parent


def main(args):
    folder=ROOT/'results'/args.experiment
    dataset=load_config(folder/'config.yaml')['dataset']
    annotation_path=ROOT/'code'/dataset['json_folder']/(dataset['test']+'.json')
    image_folder=ROOT/'code'/dataset['img_folder']
    annotations=json.loads(annotation_path.read_text())
    images={im['id']:im for im in annotations['images']}
    categories={c['id']:c['name'] for c in annotations['categories']}
    detections=defaultdict(list)
    for pred in json.loads((folder/'eval_results.json').read_text()):
        if pred['score'] >= .5:
            detections[pred['image_id']].append(pred)
    selected=[int(path.stem) for path in sorted((folder/'viz').glob('*.png'))[:12]]
    if not selected: selected=sorted(images)[:12]
    palette=gen_distinct_colors(len(categories))
    (folder/'viz').mkdir(exist_ok=True)
    for image_id in selected:
        img=pil_to_tensor(Image.open(image_folder/images[image_id]['file_name']).convert('RGB'))
        preds=detections[image_id]
        if preds:
            boxes=torch.tensor([p['bbox'] for p in preds]);boxes[:,2:]+=boxes[:,:2]
            labels=[f"{categories[p['category_id']]} {p['score']:.2f}" for p in preds]
            colors=[palette[p['category_id']-1] for p in preds]
            img=draw_bounding_boxes(img,boxes,labels,colors,width=2,font_size=16,
                 font=str(Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans.ttf'))
        save_image(img.float()/255,folder/'viz'/f'{image_id:06d}.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('experiment')
    main(parser.parse_args())
