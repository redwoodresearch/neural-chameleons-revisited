"""A doing-X-while-hiding launch refuses to write over outputs a launch wrote before, unless --overwrite."""

import types

import pytest

M = pytest.importorskip("launch.assays.math_onpolicy")  # needs modal and the launch package


class FakeVolume:
    """The volume client's listdir over a fixed {folder: [file names]}; a missing folder raises as the client does."""

    def __init__(self, folders: dict[str, list[str]]):
        self.folders = folders

    def listdir(self, folder: str):
        import modal

        if folder not in self.folders:
            raise modal.exception.NotFoundError(folder)
        return [types.SimpleNamespace(path=f"{folder}/{name}") for name in self.folders[folder]]


@pytest.fixture
def volume(monkeypatch):
    monkeypatch.setattr(M.R, "volume", FakeVolume({"runs/x-mathchat/none": ["seed201.json", "base.json"]}))


def test_an_output_written_before_is_refused(volume):
    with pytest.raises(SystemExit, match="1 of this launch's outputs exist"):
        M._refuse_overwrite(["runs/x-mathchat/none/seed201.json", "runs/x-mathchat/none/seed202.json"], False)


def test_new_outputs_pass_and_overwrite_is_on_purpose(volume):
    M._refuse_overwrite(["runs/x-mathchat/none/seed202.json", "runs/new-folder/none/base.json"], False)
    M._refuse_overwrite(["runs/x-mathchat/none/seed201.json"], True)
