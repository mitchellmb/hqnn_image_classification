import yaml
import os

config_loc = os.path.join(os.path.dirname(__file__), 'config.yml')

class Config:
    '''
    Loads config.yml as a Singleton in order to use yaml to set app configurations.
    Default values are allowed via passing through .get().
    '''

    _config=None

    @classmethod
    def _load_yaml(cls):
        if not cls._config:
            with open(config_loc, 'r') as f:
                cls._config = yaml.safe_load(f)
        return cls._config
    
    @classmethod
    def get(cls, key, default_value=None):
        return cls._load_yaml().get(key, default_value)

