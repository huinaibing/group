from .envs.explorerConf import ExplorerConf
from gymnasium.envs.registration import register

register(
    id='mars-explorer-v1',
    entry_point='mars_explorer.envs.explorerConf:ExplorerConf',
)
