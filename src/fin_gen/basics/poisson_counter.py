import numpy as np
from matplotlib import pyplot as plt

#basic poisson counter

#system parameters
k = 5
dt = 0.1 #time step
rate_const = 5.0
time_steps = np.linspace(0,20,num=200)


#20 trajectories
trajectories = []
for j in range(15):
    N_t_val = np.zeros((200,))
    for i in range(200):
        sample_one = np.random.poisson(rate_const*dt)
        N_t_val[i] = sample_one + N_t_val[i-1]
    trajectories.append(N_t_val)

for i in range(len(trajectories)):
    plt.step(time_steps,trajectories[i])
plt.show()
    
