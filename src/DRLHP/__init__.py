from gymnasium.envs import register

register(
    id='Particle-v0',
    entry_point='DRLHP.particle:Particle',
    max_episode_steps=3,  # Each episode has exactly this many steps
)
