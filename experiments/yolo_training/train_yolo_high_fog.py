"""High-fog defaults for the shared WUNET restoration and detector workflow."""

from train_yolo_with_wunet import WUNETImageRestorer, train_yolo
from train_yolo_with_wunet import main as _main


def main(argv=None):
    return _main(argv, high_fog=True)


if __name__ == '__main__':
    main()
