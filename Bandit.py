"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from logs import *
import random
import csv

logging.basicConfig
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        self.p = p
        self.rewards = []
        self.regrets = []

        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self, arm, reward):
        pass


    def experiment(self, num_trials):
        # Running an experiment with num_trials iterations
        for _ in range(num_trials):
            arm = self.pull()
            reward = Bandit_Reward[arm]
            self.rewards.append(reward)  # Recording the reward
            regret = max(Bandit_Reward) - reward  # Calculating regret
            self.regrets.append(regret)  # Recording regret
            self.update(arm, reward)  # Updating bandit's state

    def report(self, algorithm):
        # Reporting results of the experiment
        with open(f'{algorithm}_results.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Bandit', 'Reward', 'Algorithm'])
            for i in range(len(self.rewards)):
                writer.writerow([i, self.rewards[i], algorithm])
        avg_reward = sum(self.rewards) / len(self.rewards)
        avg_regret = sum(self.regrets) / len(self.regrets)
        print(f'Average Reward for {algorithm}: {avg_reward}')
        print(f'Average Regret for {algorithm}: {avg_regret}')

#--------------------------------------#



class Visualization():

    def plot1(self, epsilon_greedy_rewards, thompson_rewards):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        plt.scatter(x=range(len(epsilon_greedy_rewards)),y=epsilon_greedy_rewards, label='Epsilon Greedy')
        plt.scatter(x=range(len(thompson_rewards)),y=thompson_rewards, label='Thompson Sampling')
        plt.xlabel('Trials')
        plt.ylabel('Reward')
        plt.title('Learning Process')
        plt.legend()
        plt.show()


    def plot2(self, epsilon_greedy_rewards, thompson_rewards):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        cumulative_e_greedy_rewards = [sum(epsilon_greedy_rewards[: i + 1]) for i in range(len(epsilon_greedy_rewards))]
        cumulative_thompson_rewards = [sum(thompson_rewards[:i + 1]) for i in range(len(thompson_rewards))]

        plt.scatter(x=range(len(epsilon_greedy_rewards)), y=cumulative_e_greedy_rewards, label='Epsilon Greedy')
        plt.scatter(x=range(len(thompson_rewards)), y=cumulative_thompson_rewards, label='Thompson Sampling')
        plt.xlabel('Trials')
        plt.ylabel('Reward')
        plt.title('Cumulative Rewards')
        plt.legend()
        plt.show()



#--------------------------------------#

class EpsilonGreedy(Bandit):
    """
    """
    def __init__(self, p, initial_epsilon):
        super().__init__(p)
        self.epsilon = initial_epsilon  # Initializing epsilon
        self.q_values = [0] * len(p)  # Initializing q-values
        self.action_counts = [0] * len(p)  # Initializing action counts

    def __repr__(self):
        return 'EpsilonGreedy'

    def pull(self):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.p) - 1)
        else:
            return self.q_values.index(max(self.q_values))

    def update(self, arm, reward):
        self.action_counts[arm] += 1  # Incrementing action count
        self.q_values[arm] += (reward - self.q_values[arm]) / self.action_counts[arm]  # Updating q-value
        self.epsilon = 1 / (sum(self.action_counts) + 1)  # Decay epsilon


#--------------------------------------#

class ThompsonSampling(Bandit):
    "logg"

    def __init__(self, p, precision):
        super().__init__(p)
        self.precision = precision  # Setting precision
        self.alpha = [1.0] * len(p)  # Initializing alpha values
        self.beta = [1.0] * len(p)  # Initializing beta values

    def __repr__(self):
        return 'ThompsonSampling'

    def pull(self):
        samples = [random.betavariate(self.alpha[i], self.beta[i]) for i in range(len(self.p))]
        return samples.index(max(samples))

    def update(self, arm, reward):
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)
        if self.alpha[arm] <= 0 or self.beta[arm] <= 0:
            self.alpha[arm] = 1.0  # Resetting alpha if it becomes non-positive
            self.beta[arm] = 1.0  # Resetting beta if it becomes non-positive
        logging.debug(f'ThompsonSampling - Arm {arm} selected, Reward: {reward}')




def comparison():
    # think of a way to compare the performances of the two algorithms VISUALLY and 
    pass

if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    Bandit_Reward = [1, 2, 3, 4]
    NumberOfTrials = 20000

    epsilon_value = 0.1
    precision_value = 0.001

    epsilon_greedy_bandit = EpsilonGreedy(Bandit_Reward, epsilon_value)
    thompson_sampling_bandit = ThompsonSampling(Bandit_Reward, precision_value)

    epsilon_greedy_bandit.experiment(NumberOfTrials)
    thompson_sampling_bandit.experiment(NumberOfTrials)

    epsilon_greedy_bandit.report('epsilon_greedy')
    thompson_sampling_bandit.report('thompson_sampling')

    epsilon_greedy_rewards = epsilon_greedy_bandit.rewards
    thompson_sampling_bandit_rewards = thompson_sampling_bandit.rewards

    visualization = Visualization()
    visualization.plot1(epsilon_greedy_rewards, thompson_sampling_bandit_rewards)
    visualization.plot2(epsilon_greedy_rewards, thompson_sampling_bandit_rewards)

