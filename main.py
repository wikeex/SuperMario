import datetime
from pathlib import Path

import torch

from environment import init_resnet18_env
from mlog import MetricLogger
from mario import Mario

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")


def train(env):
    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = Mario(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir)

    logger = MetricLogger(save_dir)

    episodes = 20000
    for e in range(episodes):

        raw_state = env.reset()
        state = mario.compose_state(raw_state)

        # Play the game!
        while True:
            env.render()

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            next_raw_state, reward, done, info = env.step(action)

            next_state = mario.compose_state(next_raw_state)

            # Learn
            q, loss = mario.learn()

            # Logging
            logger.log_step(reward, loss, q)

            # Remember, 当loss值大于本批次平均loss值才去缓存，变相实现优先经验回放
            mario.cache(state, next_state, action, reward, done, loss)

            # Update state
            state = next_state

            # Check if end of game
            if done or info["flag_get"]:
                break

        if mario.curr_step > mario.burnin:
            logger.log_episode()

            if e % 20 == 0:
                logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)


if __name__ == '__main__':
    custom_env = init_resnet18_env()
    train(custom_env)
