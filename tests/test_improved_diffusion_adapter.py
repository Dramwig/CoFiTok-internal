from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "baselines" / "adapters" / "improved_diffusion"


def test_blobfile_adapter_supports_local_files(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("blobfile", ADAPTER / "blobfile.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    path = tmp_path / "nested" / "file.txt"
    with module.BlobFile(str(path), "w", encoding="utf-8") as handle:
        handle.write("ok")

    assert module.exists(str(path))
    assert module.isdir(str(path.parent))
    assert module.basename(str(path)) == "file.txt"
    assert module.listdir(str(path.parent)) == ["file.txt"]


def test_mpi4py_adapter_is_single_process() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("mpi4py", ADAPTER / "mpi4py" / "__init__.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.MPI.COMM_WORLD.Get_rank() == 0
    assert module.MPI.COMM_WORLD.Get_size() == 1
    assert module.MPI.COMM_WORLD.bcast("x", root=0) == "x"
