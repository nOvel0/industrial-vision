from pathlib import Path

import yaml


def main():
    config_path = Path(__file__).resolve().parent.parent / "configs" / "configuration.yaml"

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    print("Industrial Vision запущен")
    print(f"Камера: {config['camera']['id']}")
    print(f"Источник: {config['camera']['source']}")


if __name__ == "__main__":
    main()