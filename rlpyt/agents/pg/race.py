
from rlpyt.agents.pg.categorical import (CategoricalPgAgent,
    RecurrentCategoricalPgAgent, AlternatingRecurrentCategoricalPgAgent)
from rlpyt.models.pg.race_ff_model import RaceFfModel, RaceBasisModel


class RaceMixin:

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(image_shape=env_spaces.observation.shape,
                    output_size=5)


class RaceFfAgent(RaceMixin, CategoricalPgAgent):

    def __init__(self, ModelCls=RaceFfModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class RaceBasisAgent(RaceMixin, CategoricalPgAgent):

    def __init__(self, ModelCls=RaceBasisModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


