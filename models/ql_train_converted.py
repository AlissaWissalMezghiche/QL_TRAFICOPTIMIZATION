import traci
import numpy as np
import random
import matplotlib.pyplot as plt

SUMO_BINARY = "sumo-gui"    
SUMO_CONFIG = "../config/my_simulation.sumocfg"  
# parameters
ALPHA = 0.01       # Learning rate
GAMMA = 0.9       # Discount factor
EPSILON = 0.1     # Exploration rate
NBR_EPISODES = 100
STEPS_PER_EPISODE = 1000
# Environment parameters
STATE_SPACE_SIZE = 27   # 3×3×3=27    
ACTION_SPACE_SIZE = 2   # Transition to phase 0 (Red), Transition to phase 1 (Green)
# Initialize Q-table [rows: States; columns: Actions]
q_table = np.zeros((STATE_SPACE_SIZE, ACTION_SPACE_SIZE))

def discretize_density(density):
    if density < 4:      
        return 0         # Low 
    elif density < 6:  # 15
        return 1         # Medium 
    else:
        return 2         # High 
    
    
def discretize_speed(speed):
    if speed < 8: 
        return 0         # Slow 
    elif speed < 13: 
        return 1         # Moderate 
    else:
        return 2         # Fast 
    

def discretize_nbr(nbr):
    if nbr == 0:
        return 0
    elif nbr == 1:
        return 1
    else:
        return 1

def get_state():
    # Densité sur la voie : J11_0 (Voies du carrefour J11 entre E6 (autoroute) et E7)
    junction_density = traci.edge.getLastStepVehicleNumber(":J11_0")
    # Density on main road
    highway_e6_density = traci.edge.getLastStepVehicleNumber("E6")
    highway_e7_density = traci.edge.getLastStepVehicleNumber("E7")
    highway_density = highway_e6_density + highway_e7_density     
    # Density on ramp
    ramp_density = traci.lane.getLastStepVehicleNumber("E8_0")
    # Speed on main road
    highway_speed = (traci.edge.getLastStepMeanSpeed("E6") + traci.edge.getLastStepMeanSpeed("E7"))
    if highway_e6_density > 0 and highway_e7_density > 0:       
        highway_speed /= 2                                      
    # Discretization
    junction_density_discrete = discretize_nbr(junction_density)
    ramp_density_discrete = discretize_density(ramp_density)
    highway_speed_discrete = discretize_speed(highway_speed)
    # Get state & Calculate state index (to be used in q_table)
    state = (junction_density_discrete, ramp_density_discrete, highway_speed_discrete)
    state_index = junction_density_discrete * 9 + ramp_density_discrete * 3 + highway_speed_discrete
    
    return state, state_index



def choose_action(state_index):
    """ epsilon-greedy policy"""
    if random.uniform(0, 1) < EPSILON:
        return random.randint(0, ACTION_SPACE_SIZE - 1)     
    else:
        return np.argmax(q_table[state_index])                    
    


def take_action(action):
    
    traci.trafficlight.setPhase("J11", action) 

    
def check_collisions():
    nbr_colliding_vehicles = traci.simulation.getCollidingVehiclesNumber()
    return nbr_colliding_vehicles

def calculate_reward(state):
    ramp_density = traci.lane.getLastStepVehicleNumber("E8_0")      
    highway_speed = traci.edge.getLastStepMeanSpeed("E6")           
    nbr_colliding_vehicles = check_collisions()
    
    if ramp_density >= 6:
        reward = - 100 * nbr_colliding_vehicles - ramp_density + 0.2 * highway_speed
    else:
        reward = - 100 * nbr_colliding_vehicles - 0.2 * ramp_density + 0.2 * highway_speed
    
    return reward

def update_q_table(state_index, action, reward, next_state_index):
    best_next_action = np.argmax(q_table[next_state_index])
    q_table[state_index, action] += ALPHA * (reward + GAMMA * q_table[next_state_index, best_next_action] - q_table[state_index, action])
    

episode_rewards = []  


def q_learning(nbr_episodes = NBR_EPISODES):
    traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])

    for episode in range(NBR_EPISODES):     
        state, state_index = get_state()
        total_reward = 0
        
        for step in range(STEPS_PER_EPISODE): 
            action = choose_action(state_index)
            take_action(action)
            traci.simulationStep()
            
            next_state, next_state_index = get_state()
            reward = calculate_reward(next_state)

            
            update_q_table(state_index, action, reward, next_state_index)
            
            state = next_state
            state_index = next_state_index
            total_reward += reward
        
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
        episode_rewards.append(total_reward)  

    traci.close()

    policy = np.argmax(q_table, axis=1)
    return policy


policy_q_learning = q_learning(NBR_EPISODES)

# Plot the episode rewards
plt.plot(range(1, NBR_EPISODES + 1), episode_rewards, marker='o')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-learning Model Performance')
plt.grid(True)
# Add a legend with hyperparameter values
hyperparams_text = f'Gamma={GAMMA}, Alpha={ALPHA}, Epsilon={EPSILON}'
plt.text(NBR_EPISODES + 1, max(episode_rewards), hyperparams_text, ha='right', va='top', fontsize=10)

plt.show()