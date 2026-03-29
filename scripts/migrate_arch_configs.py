from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from gnn_archs.config import ArchConfig
from gnn_archs.util.config_migration import migrate_arch_config_dict

DEFAULT_SOURCE_DIR = Path("/home/wangjh/gnn-schedule/gen_archs/arch_config")
DEFAULT_TARGET_DIR = Path("/home/wangjh/gnn_predict/config/arch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize legacy gen_archs YAML files into the gnn_predict schema."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Legacy arch_config directory.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Output directory for normalized YAML files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.target_dir.mkdir(parents=True, exist_ok=True)

    for source_path in sorted(args.source_dir.glob("*.yaml")):
        migrated_config = migrate_file(source_path)
        target_path = args.target_dir / source_path.name
        target_path.write_text(
            yaml.safe_dump(migrated_config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(f"migrated {source_path.name} -> {target_path}")


def migrate_file(source_path: Path) -> dict:
    with source_path.open(encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)

    migrated_config = migrate_arch_config_dict(raw_config)
    ArchConfig.model_validate(migrated_config)
    return migrated_config


if __name__ == "__main__":
    main()
