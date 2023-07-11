from src.ts_train.main import hello
from src.ts_train.folder.script import method


def test_method():
    assert method() == 1


def test_hello():
    assert hello() == "Hello World!"
