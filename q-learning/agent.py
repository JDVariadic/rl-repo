class Agent:
    def __init__(self, step_size, discount_factor, epsilon_max, epsilon_min):
        self.step_size = step_size
        self.discount_factor = discount_factor
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.current_epsilon = epsilon_max

        Q = {}

        num_states = [i for i in range(16)]
        num_actions = [i for i in range(4)]

        for state in num_states:
            Q[state] = {}
            for action in num_actions:
                
                Q[state][action] = 0

        self.Q = Q

    
        