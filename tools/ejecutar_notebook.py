"""
Ejecuta el notebook de punta a punta en un kernel limpio y lo guarda con salidas.

    python tools/ejecutar_notebook.py TP2_Beltramo_Cortinas_Cura_Maragliano.ipynb

Equivale a "Reiniciar y ejecutar todo": ninguna celda puede depender de una
ejecucion anterior. Se corre desde la raiz del repositorio. Necesita nbclient e
ipykernel (pip install nbclient ipykernel). Si una celda falla, el notebook se
guarda igual, con el error visible, y el script termina con codigo 1.
"""

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def main():
    ruta = Path(sys.argv[1] if len(sys.argv) > 1
                else "TP2_Beltramo_Cortinas_Cura_Maragliano.ipynb")
    nb = nbformat.read(ruta, as_version=4)
    cliente = NotebookClient(nb, timeout=1800, kernel_name="python3",
                             resources={"metadata": {"path": str(Path.cwd())}})
    inicio = time.time()
    try:
        cliente.execute()
        print(f"EJECUTADO OK en {time.time() - inicio:.0f} s")
        return 0
    except CellExecutionError as e:
        print(f"FALLO una celda:\n{str(e)[:2000]}")
        return 1
    finally:
        nbformat.write(nb, ruta)


if __name__ == "__main__":
    sys.exit(main())
