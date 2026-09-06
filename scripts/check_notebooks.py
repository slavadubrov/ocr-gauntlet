"""Execute default notebook cells in-process, with external sockets denied."""

import json
import socket
from pathlib import Path
from unittest.mock import patch


def deny_network(*args, **kwargs):
    raise RuntimeError("Network disabled during notebook smoke check")


if __name__ == "__main__":
    for path in sorted(Path("notebooks").glob("*.ipynb")):
        namespace = {"display": lambda value: None}
        with (
            patch.object(socket.socket, "connect", deny_network),
            patch.object(socket, "create_connection", deny_network),
        ):
            for cell in json.loads(path.read_text())["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), str(path), "exec"), namespace)
        print(f"PASS {path}")
