import random
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import cv2
import imageio

# Создаем среду Taxi-v3 с рендерингом в виде массива
env = gym.make('Taxi-v3', render_mode='rgb_array')

# Параметры обучения
alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
num_episodes = 100000
max_steps = 20

# Дополнительные параметры
penalty_for_idle = -10
bonus_for_efficiency = 100
penalty_for_wrong_action = -50
penalty_per_wrong_step = -50  # Штраф за каждый "неправильный" шаг

# Путь для сохранения Q-таблицы и GIF
q_table_path = "q_table.npy"
gif_path = "taxi_training.gif"

# Инициализация Q-таблицы
if os.path.exists(q_table_path):
    print("Загружаем существующую Q-таблицу...")
    q_table = np.load(q_table_path)
else:
    print("Создаем новую Q-таблицу...")
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Список для хранения наград
episode_rewards = []

# Список для сохранения кадров для GIF
frames = []

# Функция выбора действия
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state, :])

# Функция кастомного сброса среды
def custom_reset(env):
    taxi_row = random.randint(0, 4)
    taxi_col = random.randint(0, 4)
    passenger_location = random.randint(0, 3)
    destination = random.randint(0, 3)

    # Запрещаем совпадение мест
    while destination == passenger_location:
        destination = random.randint(0, 3)

    custom_state = env.encode(taxi_row, taxi_col, passenger_location, destination)
    env.reset()
    env.unwrapped.s = custom_state
    return custom_state

# Функция рендера с наложением информации
def render_with_overlays(env, steps, total_reward, reward):
    try:
        img = env.render()  # Получаем изображение
        if img is None or len(img.shape) < 3:
            print("Ошибка рендеринга: изображение не получено")
            return None
        # Накладываем текст с информацией
        cv2.putText(img, f"Steps: {steps}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"Total Reward: {total_reward}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"Current Reward: {reward}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('Taxi-v3 Visualization', img)  # Отображаем изображение
        cv2.waitKey(1)  # Даем время на обновление окна
        return img.copy()  # Возвращаем копию изображения для сохранения
    except Exception as e:
        print(f"Ошибка рендеринга: {e}")
        return None

# Обучение агента
print("Начало обучения...")
for episode in range(num_episodes):
    state = custom_reset(env)
    done = False
    total_reward = 0
    steps = 0
    last_action = None

    for step in range(max_steps):
        steps += 1
        action = choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Доп. логика наград и штрафов
        if last_action is not None and action == last_action:
            reward += penalty_for_idle

        # Штраф за каждый "неправильный" шаг (если награда не положительная)
        if reward <= 0:
            reward += penalty_per_wrong_step

        if reward == -10:
            reward = penalty_for_wrong_action

        total_reward += reward
        last_action = action

        # Q-обновление
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state, :])
        q_table[state, action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state

        # Визуализация и сохранение кадров для GIF с 900-го эпизода
        if episode >= 99990:
            frame = render_with_overlays(env, steps, total_reward, reward)
            if frame is not None:
                frames.append(frame)
            time.sleep(0.1)  # Уменьшенная задержка для плавности

        if done:
            if steps < 15 and reward > 0:
                total_reward += bonus_for_efficiency
            break

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    episode_rewards.append(total_reward)

    if episode % 1000 == 0:
        print(f"Эпизод {episode}, суммарная награда: {total_reward}")

# Сохранение Q-таблицы
np.save(q_table_path, q_table)
print(f"Q-таблица сохранена в {q_table_path}")

# Сохранение GIF
if frames:
    imageio.mimsave(gif_path, frames, fps=10)  # Сохраняем кадры как GIF с 10 кадрами в секунду
    print(f"GIF сохранен как {gif_path}")
else:
    print("Нет кадров для создания GIF. Убедитесь, что визуализация работает с 900-го эпизода.")

# График обучения
plt.figure(figsize=(10, 6))
plt.plot(range(num_episodes), episode_rewards, label='Суммарная награда за эпизод', color='blue')
plt.xlabel('Номер эпизода')
plt.ylabel('Суммарная награда')
plt.title('Прогресс обучения агента Taxi-v3 (2D)')
plt.legend()
plt.grid(True)
plt.savefig('taxi_training_progress_2d.png')
plt.show()

env.close()
cv2.destroyAllWindows()