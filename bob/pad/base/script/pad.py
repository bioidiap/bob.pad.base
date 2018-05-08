"""The main entry for bob.pad (click-based) scripts.
"""
import click
import pkg_resources
from click_plugins import with_plugins


@with_plugins(pkg_resources.iter_entry_points('bob.pad.cli'))
@click.group()
def pad():
    """Presentation Attack Detection related commands."""
    pass
