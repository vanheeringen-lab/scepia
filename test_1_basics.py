import scepia
import pytest

def test_basic():
    cfg = genomepy.functions.config
    print(cfg)
    assert 3 == len(cfg.keys())
