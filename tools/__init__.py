# tools/__init__.py
# Keep this file minimal - do NOT re-export functions here.
# Importing 'from .check_inventory import check_inventory' would shadow the
# submodule (tools.check_inventory becomes the function, not the module),
# which breaks monkeypatch path-patching in tests.
# Each consumer should import directly from the submodule, e.g.:
#   from tools.check_inventory import check_inventory
