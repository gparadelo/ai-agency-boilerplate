import yaml
from functools import lru_cache

class ClientConfig(dict):
    pass

@lru_cache
def load_client_config(client_id: str) -> ClientConfig:
    """Load per-client YAML config."""
    path = f"configs/{client_id}.yaml"
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return ClientConfig(data)
