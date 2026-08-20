from pathlib import Path
from launcher.tk_app import main


if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve().parent))
