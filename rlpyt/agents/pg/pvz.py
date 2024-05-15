

from rlpyt.agents.pg.categorical import (CategoricalPgAgent,
    RecurrentCategoricalPgAgent, AlternatingRecurrentCategoricalPgAgent)
from rlpyt.models.pg.pvz_ff_model import pvzFfModel, \
    pvzBasisModel


class pvzMixin:

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(image_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)


class pvzFfAgent(pvzMixin, CategoricalPgAgent):

    def __init__(self, ModelCls=pvzFfModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

class pvzBasisAgent(pvzMixin, CategoricalPgAgent):

    def __init__(self, ModelCls=pvzBasisModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
