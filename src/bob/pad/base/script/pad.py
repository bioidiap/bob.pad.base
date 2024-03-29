"""The main entry for bob.pad.
"""
import click
import pkg_resources

from clapp.click import AliasedGroup
from click_plugins import with_plugins


@with_plugins(pkg_resources.iter_entry_points("bob.pad.cli"))
@click.group(cls=AliasedGroup)
def pad():
    """Presentation Attack Detection related commands."""
    pass
