import importlib.metadata

def load_plugins():
    plugins = {}
    for ep in importlib.metadata.entry_points(group="autonomous.plugins"):
        plugin_cls = ep.load()
        plugins[ep.name] = plugin_cls()
    return plugins
