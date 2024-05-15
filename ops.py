
def get_agent_cls_lunar(agent_type, algo="ppo"):
    """
    Get agent wrapper for cartpole
    """
    if agent_type in ["equivariant", "nullspace", "random"]:
        from rlpyt.agents.pg.lunarlander import LunarBasisAgent
        return LunarBasisAgent, agent_type
    else:
        from rlpyt.agents.pg.lunarlander import LunarFfAgent
        return LunarFfAgent, None



def get_agent_cls_race(agent_type, algo="ppo"):
    """
    Get agent wrapper for grid world
    """
    if agent_type in ["equivariant", "nullspace", "random"]:
        from rlpyt.agents.pg.race import RaceBasisAgent
        return RaceBasisAgent, agent_type
    elif agent_type == "cnn":
        from rlpyt.agents.pg.race import RaceFfAgent
        return RaceFfAgent, None
    else:
        raise TypeError("No agent of type {agent_type} known")



def get_agent_cls_pvz(agent_type, algo="ppo"):
    """
    Get agent wrapper for cartpole
    """
    if agent_type in ["equivarant", "nullspace", "random"]:
        from rlpyt.agents.pg.pvz import pvzBasisAgent
        return pvzBasisAgent, agent_type
    else:
        from rlpyt.agents.pg.pvz import pvzFfAgent
        return pvzFfAgent, None
