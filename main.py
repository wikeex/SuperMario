import datetime
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from environment import init_resnet18_env
from mlog import MetricLogger
from mario import Mario

use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")


def show(img):
    plt.axis("off")
    plt.imshow(img.squeeze())
    plt.show()


def train(env):
    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)

    mario = Mario(state_dim=(3, 224, 224), action_dim=env.action_space.n, save_dir=save_dir)

    logger = MetricLogger(save_dir)

    episodes = 20000
    for e in range(episodes):

        state = env.reset()
        vision_state = torch.tensor(state.__array__()).cuda()
        state = mario.vision_net(vision_state.unsqueeze(0)).detach()

        # Play the game!
        while True:
            env.render()

            # Run agent on the state
            action = mario.act(state)

            # Agent performs action
            raw_next_state, reward, done, info = env.step(action)

            # Learn
            vision_state = torch.tensor(raw_next_state.__array__()).cuda()
            next_state = mario.vision_learn(vision_state.unsqueeze(0))
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

        logger.log_episode()

        if e % 200 == 0:
            logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
            vision_state = torch.tensor(raw_next_state.__array__()).cuda()
            _, result = mario.vision_net(vision_state.unsqueeze(0), is_training=True)
            result = result.detach().cpu().numpy().squeeze(0)
            result = np.transpose(result, (1, 2, 0))
            show(result)

            show(np.transpose(raw_next_state, (1, 2, 0)))


if __name__ == '__main__':
    custom_env = init_resnet18_env()
    train(custom_env)
