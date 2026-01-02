"""
CLI commands for mbo_db.

adds `mbo db` subcommand group to the main CLI.
"""

import click

from mbo_utilities import log

logger = log.get("db.cli")


@click.group("db")
def db_cli():
    """Dataset database management commands."""


@db_cli.command("scan")
@click.argument("path", type=click.Path(exists=True))
@click.option("--recursive/--no-recursive", "-r", default=True, help="Scan subdirectories")
@click.option("--verbose", "-v", is_flag=True, help="Show verbose output")
def scan_cmd(path: str, recursive: bool, verbose: bool):
    """Scan a directory and index datasets."""
    from mbo_utilities.db.database import scan_directory

    click.echo(f"Scanning {path}...")

    def progress(current, total, p):
        if verbose and current % 100 == 0:
            click.echo(f"  [{current}/{total}] {p.name}")

    count = scan_directory(path, recursive=recursive, progress_callback=progress)
    click.secho(f"Indexed {count} datasets", fg="green")


@db_cli.command("list")
@click.option("--pipeline", "-p", help="Filter by pipeline (e.g. suite2p, zarr)")
@click.option("--status", "-s", help="Filter by status (raw, registered, segmented)")
@click.option("--tag", "-t", help="Filter by tag")
@click.option("--search", "-q", help="Search in path/name")
@click.option("--limit", "-n", default=50, help="Max results")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def list_cmd(pipeline, status, tag, search, limit, as_json):
    """List indexed datasets."""
    from mbo_utilities.db.database import get_datasets
    from mbo_utilities.db.models import DatasetStatus

    status_enum = DatasetStatus(status) if status else None
    datasets = get_datasets(
        pipeline=pipeline,
        status=status_enum,
        tag=tag,
        search=search,
        limit=limit,
    )

    if not datasets:
        click.echo("No datasets found")
        return

    if as_json:
        import json
        click.echo(json.dumps([d.to_dict() for d in datasets], indent=2))
        return

    # table output
    click.echo(f"{'ID':>5} {'Pipeline':<12} {'Status':<12} {'Size':>10} {'Name'}")
    click.echo("-" * 70)
    for d in datasets:
        click.echo(
            f"{d.id:>5} {d.pipeline:<12} {d.status.value:<12} {d.size_human:>10} {d.name}"
        )
    click.echo(f"\nTotal: {len(datasets)} datasets")


@db_cli.command("info")
@click.argument("dataset_id", type=int)
def info_cmd(dataset_id: int):
    """Show detailed info for a dataset."""
    from mbo_utilities.db.database import get_dataset, get_children, get_parents

    dataset = get_dataset(dataset_id)
    if not dataset:
        click.secho(f"Dataset {dataset_id} not found", fg="red")
        return

    click.echo(f"\n{'='*60}")
    click.echo(f"Dataset #{dataset.id}: {dataset.name}")
    click.echo(f"{'='*60}\n")

    click.echo(f"  Path:      {dataset.path}")
    click.echo(f"  Pipeline:  {dataset.pipeline}")
    click.echo(f"  Category:  {dataset.category}")
    click.echo(f"  Status:    {dataset.status.value}")
    click.echo(f"  Size:      {dataset.size_human}")
    click.echo(f"  Modified:  {dataset.modified_at}")
    click.echo(f"  Scanned:   {dataset.scanned_at}")

    if dataset.shape:
        click.echo(f"\n  Shape:     {dataset.shape}")
    if dataset.dtype:
        click.echo(f"  Dtype:     {dataset.dtype}")
    if dataset.num_frames:
        click.echo(f"  Frames:    {dataset.num_frames}")
    if dataset.num_zplanes:
        click.echo(f"  Z-planes:  {dataset.num_zplanes}")
    if dataset.num_rois:
        click.echo(f"  ROIs:      {dataset.num_rois}")

    if dataset.dx or dataset.dy or dataset.dz:
        click.echo(f"\n  Voxel size: ({dataset.dx}, {dataset.dy}, {dataset.dz}) µm")
    if dataset.fs:
        click.echo(f"  Frame rate: {dataset.fs} Hz")

    if dataset.tags:
        click.echo(f"\n  Tags:      {', '.join(dataset.tag_list)}")
    if dataset.notes:
        click.echo(f"  Notes:     {dataset.notes}")

    # show relationships
    parents = get_parents(dataset.id)
    children = get_children(dataset.id)

    if parents:
        click.echo("\n  Derived from:")
        for p in parents:
            click.echo(f"    - [{p.id}] {p.name}")

    if children:
        click.echo("\n  Processed to:")
        for c in children:
            click.echo(f"    - [{c.id}] {c.name}")

    click.echo()


@db_cli.command("link")
@click.argument("source_id", type=int)
@click.argument("target_id", type=int)
@click.option("--type", "link_type", default="processed_from", help="Link type")
def link_cmd(source_id: int, target_id: int, link_type: str):
    """Link two datasets (source -> target)."""
    from mbo_utilities.db.database import link_datasets, get_dataset

    source = get_dataset(source_id)
    target = get_dataset(target_id)

    if not source:
        click.secho(f"Source dataset {source_id} not found", fg="red")
        return
    if not target:
        click.secho(f"Target dataset {target_id} not found", fg="red")
        return

    link_datasets(source_id, target_id, link_type)
    click.secho(
        f"Linked: {source.name} -> {target.name} ({link_type})",
        fg="green"
    )


@db_cli.command("tag")
@click.argument("dataset_id", type=int)
@click.argument("tag")
@click.option("--remove", "-r", is_flag=True, help="Remove tag instead of adding")
def tag_cmd(dataset_id: int, tag: str, remove: bool):
    """Add or remove a tag from a dataset."""
    from mbo_utilities.db.database import add_tag, remove_tag, get_dataset

    dataset = get_dataset(dataset_id)
    if not dataset:
        click.secho(f"Dataset {dataset_id} not found", fg="red")
        return

    if remove:
        remove_tag(dataset_id, tag)
        click.echo(f"Removed tag '{tag}' from {dataset.name}")
    else:
        add_tag(dataset_id, tag)
        click.echo(f"Added tag '{tag}' to {dataset.name}")


@db_cli.command("stats")
def stats_cmd():
    """Show database statistics."""
    from mbo_utilities.db.database import get_stats, get_db_path

    stats = get_stats()

    click.echo(f"\nDatabase: {get_db_path()}")
    click.echo(f"\nTotal datasets: {stats['total_datasets']}")

    # format total size
    size = stats["total_size_bytes"]
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            click.echo(f"Total size: {size:.1f} {unit}")
            break
        size /= 1024

    if stats["by_pipeline"]:
        click.echo("\nBy pipeline:")
        for pipeline, count in sorted(stats["by_pipeline"].items()):
            click.echo(f"  {pipeline}: {count}")

    if stats["by_status"]:
        click.echo("\nBy status:")
        for status, count in sorted(stats["by_status"].items()):
            click.echo(f"  {status}: {count}")

    click.echo()


@db_cli.command("clear")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
def clear_cmd(yes: bool):
    """Clear all datasets from the database."""
    from mbo_utilities.db.database import get_db_path

    if not yes:
        click.confirm("This will delete all indexed datasets. Continue?", abort=True)

    db_path = get_db_path()
    if db_path.exists():
        db_path.unlink()
        click.secho("Database cleared", fg="green")
    else:
        click.echo("Database does not exist")


@db_cli.command("browse")
def browse_cmd():
    """Launch the dataset browser GUI."""
    from mbo_utilities.db.gui import launch_browser
    launch_browser()


@db_cli.command("open")
@click.argument("dataset_id", type=int)
def open_cmd(dataset_id: int):
    """Open a dataset in the MBO viewer."""
    from mbo_utilities.db.database import get_dataset
    from mbo_utilities.gui.run_gui import run_gui

    dataset = get_dataset(dataset_id)
    if not dataset:
        click.secho(f"Dataset {dataset_id} not found", fg="red")
        return

    click.echo(f"Opening {dataset.name}...")
    run_gui(dataset.path)
