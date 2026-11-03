"""Restore the exact classifier weights from the public release, checking SHA-256."""
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent.parent
URL = 'https://github.com/matejpopovski/bmi771-assignment2/releases/download/classification-v1/classification-checkpoints.zip'


def main():
    audit = json.loads((ROOT/'results/classification_audit.json').read_text())
    if not audit.get('ready'):
        raise RuntimeError('Update the repository to the completed classifier contribution first')
    expected = {f'results/{name}/model_final.pth.tar': info['checkpoint_sha256']
                for name, info in audit['experiments'].items()}
    missing=[]
    for name, digest in expected.items():
        target=ROOT/name
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
                raise FileExistsError(f'{target} contains different weights; move it aside first')
        else:
            missing.append(name)
    if not missing:
        print('All four classifier checkpoints already match the audited hashes.')
        return
    with tempfile.TemporaryDirectory() as directory:
        archive=Path(directory)/'weights.zip'
        urllib.request.urlretrieve(URL,archive)
        with ZipFile(archive) as z:
            if set(z.namelist())!=set(expected):
                raise RuntimeError('Unexpected archive contents')
            for name in missing:
                data=z.read(name)
                if hashlib.sha256(data).hexdigest()!=expected[name]:
                    raise RuntimeError('Checksum mismatch: '+name)
                target=ROOT/name
                target.parent.mkdir(parents=True,exist_ok=True)
                temp=target.with_suffix('.download')
                temp.write_bytes(data)
                temp.replace(target)
    print('Downloaded and verified all missing classifier checkpoints.')


if __name__ == '__main__':
    main()
