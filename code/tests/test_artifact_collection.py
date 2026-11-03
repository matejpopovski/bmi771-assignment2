from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_experiments import collect


def test_collection_replaces_old_events_and_excludes_intermediate_weights(tmp_path):
    source, destination = tmp_path/'run', tmp_path/'result'
    (source/'logs').mkdir(parents=True)
    (destination/'logs').mkdir(parents=True)
    (source/'logs/new-event').write_text('new')
    (destination/'logs/old-event').write_text('old')
    (source/'epoch_004.pth.tar').write_text('intermediate')
    (source/'model_final.pth.tar').write_text('final')
    collect(source, destination)
    assert sorted(p.name for p in (destination/'logs').iterdir()) == ['new-event']
    assert (destination/'model_final.pth.tar').read_text() == 'final'
    assert not (destination/'epoch_004.pth.tar').exists()
