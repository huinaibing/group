from rlpyt.agents.pg.categorical import (CategoricalPgAgent,
                                         RecurrentCategoricalPgAgent, AlternatingRecurrentCategoricalPgAgent)

from rlpyt.models.pg.lunar_ff_model import LunarFfModel, \
    LunarBasisModel


class LunarMixin:

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(image_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)


class LunarFfAgent(LunarMixin, CategoricalPgAgent):

    def __init__(self, ModelCls=LunarFfModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


class LunarBasisAgent(LunarMixin, CategoricalPgAgent):

    def __init__(self, ModelCls=LunarBasisModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
