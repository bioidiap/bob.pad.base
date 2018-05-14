"""The main entry for bob.vuln
"""
import click
import pkg_resources
from click_plugins import with_plugins

@with_plugins(pkg_resources.iter_entry_points('bob.vuln.cli'))
@click.group()
def vuln():
  """Presentation Attack Detection related commands."""
  pass
