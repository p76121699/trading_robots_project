import numpy as np
from tqdm import tqdm
from argparse import Namespace
from StockTradingEnv import StockTradingEnvMulti
from algs.drl_agent import ReinforceAgent
from config.algs.reinforce_config import REINFORCE_CONFIG
from config.env.env_config import env_config, test_env_config
from config.training_config import training_config

def pad_state_sequence(state_seq, window_size):
    T, D = state_seq.shape
    if T == window_size:
        return state_seq
    elif T < window_size:
        pad_len = window_size - T
        pad = np.zeros((pad_len, D), dtype=state_seq.dtype)
        return np.vstack([pad, state_seq])
    else:
        return state_seq[-window_size:]  # truncate

def train(args):
    env = StockTradingEnvMulti(Namespace(**env_config))

    agent = ReinforceAgent(Namespace(**REINFORCE_CONFIG))
    
    episode_steps = len(env.dates) - 49

    # 日誌資料
    logs = {
        "episode_rewards": [],
        "cash_history": [],
        "position_history": [],
        "actor_loss_history": [],
        "step_rewards": [],
        "cumulative_return": [],
    }

    print("Starting training...")

    for episode in range(args.num_episodes):
        state = env.reset()
        total_reward, done, step_count = 0.0, False, 0
        state = env._get_state()
        pbar = tqdm(total=episode_steps, desc=f"Episode {episode+1}", leave=False)

        actor_losses, episode_step_rewards = [], []

        while not done:
            pad_state = pad_state_sequence(state, env.window_size)
            
            action, logits = agent.select_action(pad_state)
            next_state, reward, done, result = env.step(action)

            # 儲存 transition
            agent.store_transition(logits, reward)
            
            logs["step_rewards"].append(reward)
            episode_step_rewards.append(reward)

            # 更新狀態與 epsilon
            state = next_state
            total_reward += reward
            pbar.update(1)
            pbar.set_postfix({"Reward": f"{total_reward:.2f}"})

        actor_loss = agent.update()
        actor_losses.append(actor_loss)
        pbar.close()
        cash_balance = env.cash_balance + (env.positions * env.close_mat[env.day_ptr - 1]).sum()
        cumulative_return = (cash_balance - env.initial_cash) / env.initial_cash

        logs["cumulative_return"].append(cumulative_return)
        logs["episode_rewards"].append(total_reward)
        logs["cash_history"].append(cash_balance)
        logs["position_history"].append(np.sum(env.positions))
        logs["actor_loss_history"].append(np.mean(actor_losses))

        print(f"Episode {episode+1} finished. Total reward: {total_reward:.2f}, mean_reward: {np.mean(episode_step_rewards):.2f}, cash: {env.cash_balance}, position: {np.sum(env.positions)}, cumulative return: {cumulative_return}")

    for k, v in logs.items():
        np.save(f"./stats/{k}.npy", np.array(v))
    agent.save_model("./models/reinforce_model")

def test():
    env = StockTradingEnvMulti(Namespace(**test_env_config))
    
    agent = ReinforceAgent(Namespace(**REINFORCE_CONFIG))
    
    agent.load_model("./models/reinforce_model")

    print("Starting testing...")

    episode_steps = len(env.dates) - 49
    
    state = env.reset()
    total_reward, done, step_count = 0.0, False, 0
    state = env._get_state()
    pbar = tqdm(total=episode_steps, leave=False)

    while not done:
        pad_state = pad_state_sequence(state, env.window_size)
            
        action, _ = agent.select_action(pad_state)

        next_state, reward, done, _ = env.step(action)

        state = next_state
        total_reward += reward
        step_count += 1

        pbar.update(1)
        pbar.set_postfix({"Reward": f"{total_reward:.2f}"})

    pbar.close()
    cash_balance = env.cash_balance + (env.positions * env.close_mat[env.day_ptr - 1]).sum()
    cumulative_return = (cash_balance - env.initial_cash) / env.initial_cash

    print(f"cumulative return: {cumulative_return}")


if __name__ == "__main__":
    args = Namespace(**training_config)
    train(args)
    test()
