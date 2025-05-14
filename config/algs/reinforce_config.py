REINFORCE_CONFIG = {
    "actor_lr": 1e-6,
    "state_dim": 7 * 20,  # 7 features * 20 stocks
    "action_dim": 11,     # discrete actions: -50, -40, ..., 0, ..., +50
    "time_seq": 30,
    "n_step": 8,
    "gamma": 0.99
}
