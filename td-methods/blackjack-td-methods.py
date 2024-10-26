import os
from math import floor
from collections.abc import Callable
from collections import deque
from typing import Mapping, Any
import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


# ------------------------------- Type Aliases --------------------------------
type State = tuple[int, int, int]
type Action = int
type Policy = Callable[[State], Action]


# ----------------------------- Training Params ------------------------------
num_episodes = int(os.environ.get("NUM_EPS", "100_000"))

# Minimum value epsilon should fall to
epsilon_min = 1e-2

# ---------------------------------- Results ----------------------------------
moving_avg_window_size = floor(num_episodes / 20)
performance_results_size = num_episodes - moving_avg_window_size


# ------------------------------- Debug Params -------------------------------
debug = bool(int(os.environ.get("DEBUG", "0")))
debug_print_period = floor(num_episodes / 10)


# env.observation_space = Tuple(Discrete(32), Discrete(11), Discrete(2))
# Is there a way to get the observation space shape dynamically?
# Consider using `FlattenObservation`
state_space_shape = (32, 11, 2)
action_space_size = 2
# action space = 2 (0 = stand, 1 = hit)
state_action_space_shape = (*state_space_shape, action_space_size)


# An example policy to find the value function for during prediction
# Simple strategy; hit if lower than 17
def some_policy(state: State) -> Action:
    current_score, dealers_card, usable_ace = state
    return 1 if current_score < 17 else 0


def epsilon_greedy_policy(
    state: State, q_values: NDArray[np.float64], epsilon: float
) -> int:
    if np.random.random() < epsilon:
        return np.random.choice(action_space_size - 1)
    else:
        score, dealers_hand, ace = state
        return np.argmax(q_values[score][dealers_hand][ace])


# Play through with a random policy to act as a control variable
def play_random_policy(
    env: gym.Env, num_episodes: int
) -> tuple[None, NDArray[np.float16]]:
    # Recording performance metrics
    moving_average = deque(maxlen=moving_avg_window_size)
    results = np.zeros(num_episodes, np.float16)

    for episode_num in range(num_episodes):
        state, _ = env.reset()
        action = np.random.choice(action_space_size - 1)
        total_episode_return = 0
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_episode_return += reward
            next_action = np.random.choice(action_space_size - 1)

            # Destructuring tuples to reduce line length
            score, dealer, ace = state
            next_score, next_dealer, next_ace = next_state

            state = next_state
            action = next_action
            done = terminated or truncated

        moving_average.append(total_episode_return)
        results[episode_num] = sum(moving_average) / len(moving_average)

    return None, results


# TD(0) prediction
def td0(
    env: gym.Env,
    policy: Policy,
    num_episodes: int,
    learning_rate: float,
    discount_factor: float,
) -> NDArray[np.float64]:
    value_func = np.zeros(state_space_shape)

    for episode_num in range(num_episodes):
        if debug and episode_num % debug_print_period == 0:
            print(f"episode: {episode_num}")

        state, _ = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            score, dealer, ace = state
            next_score, next_dealer, next_ace = next_state

            td_target = (
                reward + discount_factor * value_func[next_score][next_dealer][next_ace]
            )
            td_error = td_target - value_func[score][dealer][ace]
            value_func[score][dealer][ace] += learning_rate * td_error

            state = next_state
            done = terminated or truncated

    return value_func


# On-policy TD(0) control
def sarsa(
    env: gym.Env,
    num_episodes: int,
    learning_rate_start: float,
    learning_rate_half_life: float,
    discount_factor: float,
    epsilon_start: float,
    epsilon_half_life: float,
) -> tuple[NDArray[np.float64], NDArray[np.float16]]:
    q_values = np.zeros(state_action_space_shape)

    learning_rate = learning_rate_start
    epsilon = epsilon_start

    # Recording performance metrics
    moving_average = deque(maxlen=moving_avg_window_size)
    results = np.zeros(num_episodes, np.float16)

    for episode_num in range(num_episodes):
        if debug and episode_num > 0 and episode_num % debug_print_period == 0:
            print(
                f"episode: {episode_num}, "
                f"learning rate: {learning_rate}, "
                f"epsilon: {epsilon}, "
                f"avg score: {results[episode_num - 1]}"
            )

        state, _ = env.reset()
        action = epsilon_greedy_policy(state, q_values, epsilon)
        # Total *undiscounted* rewards gained from episode
        # Only used to report performance
        total_episode_return = 0
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_episode_return += reward
            next_action = epsilon_greedy_policy(next_state, q_values, epsilon)

            # Destructuring tuples to reduce line length
            score, dealer, ace = state
            next_score, next_dealer, next_ace = next_state

            # SARSA update
            td_target = (
                reward
                + discount_factor
                * q_values[next_score][next_dealer][next_ace][next_action]
            )
            td_error = td_target - q_values[score][dealer][ace][action]
            q_values[score][dealer][ace][action] += learning_rate * td_error

            state = next_state
            action = next_action
            done = terminated or truncated

        learning_rate = learning_rate_start * pow(
            0.5, episode_num / learning_rate_half_life
        )
        epsilon = max(
            epsilon_min, epsilon_start * pow(0.5, episode_num / epsilon_half_life)
        )
        moving_average.append(total_episode_return)
        results[episode_num] = sum(moving_average) / len(moving_average)

    return q_values, results


# Off-policy TD(0) control
def q_learning(
    env: gym.Env,
    num_episodes: int,
    learning_rate_start: float,
    learning_rate_half_life: float,
    discount_factor: float,
    epsilon_start: float,
    epsilon_half_life: float,
) -> tuple[NDArray[np.float64], NDArray[np.float16]]:
    q_values = np.zeros(state_action_space_shape)

    learning_rate = learning_rate_start
    epsilon = epsilon_start

    # Recording performance metrics
    moving_average = deque(maxlen=moving_avg_window_size)
    results = np.zeros(num_episodes, np.float16)

    for episode_num in range(num_episodes):
        if debug and episode_num > 0 and episode_num % debug_print_period == 0:
            print(
                f"episode: {episode_num}, "
                f"learning rate: {learning_rate}, "
                f"epsilon: {epsilon}, "
                f"avg score: {results[episode_num - 1]}"
            )

        state, _ = env.reset()
        # Total *undiscounted* rewards gained from episode
        # Only used to report performance
        total_episode_return = 0
        done = False

        while not done:
            action = epsilon_greedy_policy(state, q_values, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_episode_return += reward

            # Destructuring tuples to reduce line length
            score, dealer, ace = state
            next_score, next_dealer, next_ace = next_state

            # Q-learning update
            best_action = np.argmax(q_values[next_score][next_dealer][next_ace])
            # Don't use future reward if state is a terminal state
            future_q_value = (not done) * q_values[next_score][next_dealer][next_ace][
                best_action
            ]

            td_target = reward + discount_factor * future_q_value
            td_error = td_target - q_values[score][dealer][ace][action]
            q_values[score][dealer][ace][action] += learning_rate * td_error

            state = next_state

        learning_rate = learning_rate_start * pow(
            0.5, episode_num / learning_rate_half_life
        )
        epsilon = max(
            epsilon_min, epsilon_start * pow(0.5, episode_num / epsilon_half_life)
        )
        moving_average.append(total_episode_return)
        results[episode_num] = sum(moving_average) / len(moving_average)

    return q_values, results


def play_episode(env: gym.Env, policy: Policy) -> list[tuple[State, Action, int]]:
    state, _ = env.reset()
    episode = []
    done = False

    while not done:
        action = policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        episode.append((state, action, reward))

        done = terminated or truncated
        state = next_state

    return episode


def mc_prediction_every_visit(
    env: gym.Env,
    policy: Policy,
    num_episodes: int,
    discount_factor: float,
) -> NDArray[np.float64]:
    num_visits = np.zeros(state_space_shape)
    value_func = np.zeros(state_space_shape)

    for episode_num in range(num_episodes):
        if debug and episode_num > 0 and episode_num % debug_print_period == 0:
            print(f"episode: {episode_num}")

        episode = play_episode(env, policy)

        experienced_return = 0
        for timestep in range(len(episode) - 1, -1, -1):
            state, _, reward = episode[timestep]
            experienced_return = discount_factor * experienced_return + reward

            score, dealer, ace = state

            num_visits[score][dealer][ace] += 1

            # These are just to bypass the "line too long" errors
            predicted_value = value_func[score][dealer][ace]
            visits = num_visits[score][dealer][ace]

            monte_carlo_error = experienced_return - predicted_value
            value_func[score][dealer][ace] += (1 / visits) * monte_carlo_error

    return value_func


def mc_control_every_visit(
    env: gym.Env,
    num_episodes: int,
    discount_factor: float,
    epsilon_start: float,
    epsilon_half_life: float,
) -> tuple[NDArray[np.float64], NDArray[np.float16]]:
    q_values = np.full(state_action_space_shape, 0, np.float64)
    num_visits = np.zeros(state_action_space_shape, np.int64)

    epsilon = epsilon_start

    # Recording performance metrics
    moving_average = deque(maxlen=moving_avg_window_size)
    results = np.zeros(num_episodes, np.float16)

    for episode_num in range(num_episodes):
        if debug and episode_num > 0 and episode_num % debug_print_period == 0:
            print(
                f"episode: {episode_num}, "
                f"epsilon: {epsilon}, "
                f"avg score: {results[episode_num - 1]}"
            )

        def policy(state: State) -> Action:
            return epsilon_greedy_policy(state, q_values, epsilon)

        episode = play_episode(env, policy)

        experienced_return = 0

        for timestep in reversed(episode):
            state, action, reward = timestep

            score, dealer, ace = state
            experienced_return = discount_factor * experienced_return + reward

            num_visits[score][dealer][ace][action] += 1

            # Just to escape the "line too long" linting errors
            q = q_values[score][dealer][ace][action]
            visits = num_visits[score][dealer][ace][action]

            q_values[score][dealer][ace][action] += (1 / visits) * (
                experienced_return - q
            )

        epsilon = max(
            epsilon_min, epsilon_start * pow(0.5, episode_num / epsilon_half_life)
        )
        total_episode_return = sum(reward for _, _, reward in episode)
        moving_average.append(total_episode_return)
        results[episode_num] = sum(moving_average) / len(moving_average)

    return q_values, results


def plot_performance(models: list[Mapping[str, Any]]) -> None:
    fig, ax = plt.subplots()

    for model in models:
        ax.plot(model["performance_results"], label=model["label"])

    ax.set_xlim(0, num_episodes)
    ax.set_ylim(-0.25, -0.1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Score from Game")
    ax.set_title("Performance Comparison of RL Algorithms in Blackjack")
    ax.legend()
    fig.tight_layout()
    fig.savefig("blackjack_rl_comparison.png")
    plt.show()


env = gym.make("Blackjack-v1", natural=False, sab=True)


if __name__ == "__main__":
    hyperparameters = {
        "monte_carlo": {
            "discount_factor": 0.1,
            "epsilon_start": 0.1,
            "epsilon_half_life": num_episodes / 3,
        },
        "sarsa": {
            "learning_rate_start": 5e-7,
            "learning_rate_half_life": num_episodes / 2,
            "discount_factor": 0.9,
            "epsilon_start": 0.95,
            "epsilon_half_life": num_episodes / 5,
        },
        "q_learning": {
            "learning_rate_start": 1e-2,
            "learning_rate_half_life": num_episodes / 2,
            "discount_factor": 0.95,
            "epsilon_start": 0.99,
            "epsilon_half_life": num_episodes / 2,
        },
    }
    models = [
        {
            "label": "Monte Carlo",
            "algo_func": mc_control_every_visit,
            "args_list": [
                env,
                num_episodes,
                hyperparameters["monte_carlo"]["discount_factor"],
                hyperparameters["monte_carlo"]["epsilon_start"],
                hyperparameters["monte_carlo"]["epsilon_half_life"],
            ],
            "q_func": None,
            "performance_results": None,
        },
        {
            "label": "SARSA",
            "algo_func": sarsa,
            "args_list": [
                env,
                num_episodes,
                hyperparameters["sarsa"]["learning_rate_start"],
                hyperparameters["sarsa"]["learning_rate_half_life"],
                hyperparameters["sarsa"]["discount_factor"],
                hyperparameters["sarsa"]["epsilon_start"],
                hyperparameters["sarsa"]["epsilon_half_life"],
            ],
            "q_func": None,
            "performance_results": None,
        },
        {
            "label": "Q-learning",
            "algo_func": q_learning,
            "args_list": [
                env,
                num_episodes,
                hyperparameters["q_learning"]["learning_rate_start"],
                hyperparameters["q_learning"]["learning_rate_half_life"],
                hyperparameters["q_learning"]["discount_factor"],
                hyperparameters["q_learning"]["epsilon_start"],
                hyperparameters["q_learning"]["epsilon_half_life"],
            ],
            "q_func": None,
            "performance_results": None,
        },
        {
            "label": "Random",
            "algo_func": play_random_policy,
            "args_list": [env, num_episodes],
            "q_func": None,
            "performance_results": None,
        },
    ]

    for model in models:
        print(f"\nTraining: {model['label']}")
        print(
            "================================================================================"
        )
        model["q_func"], model["performance_results"] = model["algo_func"](
            *model["args_list"]
        )

        if debug and model["q_func"] is not None:
            print("Printing found Q-values (state-action values) for optimal policy...")

            for current_score in range(state_space_shape[0]):
                print(f"current score: {current_score}")
                for dealers_card in range(state_space_shape[1]):
                    print(f"\tdealer's card: {dealers_card}")
                    for usable_ace in range(state_space_shape[2]):
                        print(f"\t\tusable ace: {usable_ace}")
                        print("\t\t\taction values:")
                        print("\t\t\t(0 = stand, 1 = hit)")
                        print(
                            f"\t\t\t{model['q_func'][current_score][dealers_card][usable_ace]}"
                        )
        print(f"Final average score: {model['performance_results'][-1]}")

    print("Plotting results...")
    plot_performance(models)
