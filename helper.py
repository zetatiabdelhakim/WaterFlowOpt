import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(reward, mean_reward):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Tasks')
    plt.ylabel('Reward')
    plt.plot(reward)
    plt.plot(mean_reward)
    plt.ylim(ymin=0)
    plt.text(len(reward)-1, reward[-1], str(reward[-1]))
    plt.text(len(mean_reward)-1, mean_reward[-1], str(mean_reward[-1]))
    plt.show(block=False)
    plt.pause(.1)
