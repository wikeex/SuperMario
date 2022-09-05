import datetime
import os
import pickle
import time
from threading import Thread, Lock

import gym_super_mario_bros
import torch
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from pynput.keyboard import Listener

from environment import SkipFrame, GrayScaleObservation, ResizeObservation, custom_space

locker = Lock()
current_press_keys = set()


KEY_MAP = {'': 0, 'a': 1, 'd': 2, 'w': 3, 's': 4, 'j': 5, 'k': 6, 'aj': 7, 'ak': 8, 'dj': 9, 'dk': 10, 'jw': 11,
           'kw': 12, 'js': 13, 'ks': 14, 'jk': 15}
Cache = []


def keyboard_listener(on_press_func, on_release_func):
    with Listener(on_press=on_press_func, on_release=on_release_func) as listener:
        listener.join()


def on_press(key):
    global current_press_keys

    try:
        locker.acquire()
        current_press_keys.add(key.char)
        locker.release()
    except AttributeError:
        return


def on_release(key):
    global current_press_keys
    try:
        locker.acquire()
        current_press_keys.remove(key.char)
        locker.release()
    except AttributeError:
        return


def cache(state, next_state, action, reward, done):
    state = state.__array__()
    next_state = next_state.__array__()
    state = torch.tensor(state)
    next_state = torch.tensor(next_state)
    action = torch.tensor([action])
    reward = torch.tensor([reward])
    done = torch.tensor([done])

    Cache.append((state, next_state, action, reward, done,))


def save(filename):
    with open(f'{filename}.pkl', 'wb') as f:
        pickle.dump(Cache, f, pickle.HIGHEST_PROTOCOL)
    print(f'数据已保存，state数量：{len(Cache)}')


def load(dirname):
    filenames = os.listdir(dirname)
    total_data = []
    for filename in filenames:
        if filename.startswith('mario_master_data') and filename.endswith('.pkl'):
            with open(os.path.join(dirname, filename), 'rb') as f:
                data = pickle.load(f)
                total_data.extend(data)
    return total_data


def play():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, custom_space)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    state = env.reset()
    while True:
        env.render()

        locker.acquire()
        action = KEY_MAP.get(''.join(sorted(list(current_press_keys))), 0)
        locker.release()

        next_state, reward, done, info = env.step(action)
        cache(state, next_state, action, reward, done)
        state = next_state

        if done or info['flag_get'] is True:
            while True:
                choice = input(f'需要保存本批次数据吗？输入y或者n并按回车\n')
                if choice == 'y':
                    save(f"mario_master_data_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")
                    print('已保存\n')
                    break
                elif choice == 'n':
                    print('放弃保存\n')
                    break
                else:
                    print('输入有误，请重新输入\n')
                    continue
            break
        time.sleep(0.044)
    env.close()


if __name__ == '__main__':
    read_key_thread = Thread(target=keyboard_listener, args=(on_press, on_release))
    read_key_thread.start()
    play()
