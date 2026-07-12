"""Single-process mpi4py subset for improved-diffusion baseline runs."""

from __future__ import annotations


class _SingleProcessComm:
    rank = 0
    size = 1

    def Get_rank(self) -> int:
        return self.rank

    def Get_size(self) -> int:
        return self.size

    def bcast(self, value, root: int = 0):  # noqa: ANN001, ANN201 - mpi4py-compatible
        if root != 0:
            raise ValueError("single-process MPI stub only supports root=0")
        return value

    def Barrier(self) -> None:
        return None

    def barrier(self) -> None:
        return None


class _MPI:
    COMM_WORLD = _SingleProcessComm()


MPI = _MPI()
