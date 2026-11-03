"""Re-evaluate saved classifier artifacts and retain a separate verification record."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import time
import torch
from cls_libs import load_config, build_model, build_loader, evaluate
from runtime_utils import seed_everything

ROOT=Path(__file__).resolve().parent.parent


def main(args):
    os.chdir(ROOT/'code')
    seed_everything(771)
    device=torch.device(args.device)
    for name in args.experiments:
        folder=ROOT/'results'/name
        config=load_config(folder/'config.yaml')
        config['devices']=[args.device]
        config['loader'].update(batch_size=128,num_workers=args.workers)
        checkpoint_path=folder/'model_final.pth.tar'
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
        model=build_model(config,pretrained=False)
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        model.to(device)
        loader=build_loader(config,False)
        start=time.time()
        metrics=evaluate(loader,model,checkpoint['epoch'],device,eval_aug=False,
                         print_freq=0,return_metrics=True)
        metrics['seconds']=time.time()-start
        primary=json.loads((folder/'metrics.json').read_text())
        delta={key:metrics[key]-primary[key] for key in ('top1','top5')}
        result={'device':args.device,'torch':torch.__version__,'batch_size':128,
                'metrics':metrics,'difference_percentage_points':delta,
                'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()}
        (folder/'verification.json').write_text(json.dumps(result,indent=2))
        if metrics['num_images']!=10000 or any(abs(x)>.1 for x in delta.values()):
            raise RuntimeError(f'{name}: re-evaluation disagrees with original metrics: {result}')
        print(name,json.dumps(result),flush=True)
        del loader,model,checkpoint


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='mps')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--experiments',nargs='+',choices=['simple','cnn','vit','resnet18'],default=['simple','cnn','vit','resnet18'])
    main(parser.parse_args())
