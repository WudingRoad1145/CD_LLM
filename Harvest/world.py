from pprint import pprint, pformat
import pandas as pd
from collections import OrderedDict
from agent import *
from apple import *

class WorldError(Exception):
    """
    Exception raised for errors in the world. To indicate that the world is in an invalid state.
    """
    pass

class World:
    def __init__(self, x_size, y_size, num_apples=20):
        self.x_size = x_size
        self.y_size = y_size
        self.map = [[[] for _ in range(x_size)] for _ in range(y_size)]
        self.instances = {}
        self.agents_map = OrderedDict()
        self.name_to_id = {}
        self.global_id = 0
        self.contract_active = False
        # Randomly distribute apples
        for _ in range(num_apples):
            x, y = np.random.randint(0, x_size), np.random.randint(0, y_size)
            while self.is_occupied(x, y):
                x, y = np.random.randint(0, x_size), np.random.randint(0, y_size)
            self.add_instance(Apple(x, y))

    def is_occupied(self, x, y):
        if x < 0 or x >= self.x_size or y < 0 or y >= self.y_size:
            return True
        
        id_list = self.map[y][x]
        for id in id_list:
            if self.instances[id] != []: # might wrong
                return True
        return False

    def exists(self, id):
        """
        Check if the instance exists in the world.
        """
        return id in self.instances

    def get_position(self, id):
        instance = self.instances[id]
        return instance.x, instance.y

    def get_instance_by_id(self, id):
        return self.instances[id]

    def get_instance_by_name(self, name):
        return self.instances[self.name_to_id[name.lower()]]
    
    def get_all_instances(self):
        """return list of all instance(class) for prompting"""
        return self.instances.values()

    def get_world_state(self):
        world_state = [[[] for _ in range(self.x_size)] for _ in range(self.y_size)]
        for instance in self.instances.values():
            world_state[instance.y][instance.x].append(instance.name)
        world_state = [
            [" & ".join(i) if i != [] else "." for i in j] for j in world_state
        ]
        
        instance_details = []
        for instance in self.instances.values():
            instance_details.append(instance.get_info())
        return world_state, instance_details

    def add_instance(self, instance):
        id = self.global_id
        if self.is_occupied(instance.x, instance.y):
            raise WorldError("Cannot add instance: the location is occupied")

        id = self.global_id
        self.instances[id] = instance
        self.global_id += 1
        return id

    def spawn_apples(self):
        for row in range(self.x_size):
            for col in range(self.y_size):
                if not any(isinstance(inst, Apple) for inst in self.map[col][row]):
                    nearby_apples = self.count_nearby_apples(row, col)
                    if nearby_apples == 0:
                        continue
                    spawn_prob = [0, 0.005, 0.02, 0.05][min(nearby_apples, 3)]
                    if np.random.random() < spawn_prob:
                        self.add_instance(Apple(row, col))

    def count_nearby_apples(self, x, y, radius=5):
        count = 0
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                if 0 <= x+i < self.x_size and 0 <= y+j < self.y_size:
                    if any(isinstance(inst, Apple) for inst in self.map[y+j][x+i]):
                        count += 1
        return count

    def enforce_contract(self, contract_param):
        x = contract_param['x']  # The number of apples to be transferred
        
        # Check each agent to see if they violated the contract
        for agent_id, agent in self.agents_map.items():
            if agent.just_collected_apple:  # Assuming there's a flag in the agent class to track this
                nearby_apples = self.count_nearby_apples(agent.x, agent.y)
                
                # If the agent collected an apple in a low-density region
                if nearby_apples < 4:
                    # Take x apples from the violating agent
                    agent.rewards = max(0, agent.rewards - x)
                    
                    # Distribute the apples equally among the other agents
                    num_other_agents = len(self.agents_map) - 1
                    apples_per_agent = x // num_other_agents
                    for other_agent_id, other_agent in self.agents_map.items():
                        if other_agent_id != agent_id:
                            other_agent.rewards += apples_per_agent

    def agent_update_order(self, agent, order):
        #TODO
        pass

    def __repr__(self):
        w, infos = self.get_world_state()
        print(pd.DataFrame.from_records(infos).sort_values("name").drop(
            ["id", 'x', 'y'], axis=1))
        w = pd.DataFrame(w)
        idx_list = list(range(self.y_size))
        w.index = list(map(lambda s: "|" + str(s) + "|", idx_list))
        w = w.rename(columns=lambda s: "|" + str(s) + "|")
        return pformat(w, indent=4)

    def run(self, n_rounds):
        """
        Run the world for n_rounds
        """
        for _ in range(n_rounds):
            self._round()

    def _round(self):
        # 1. Randomly pick one agent to propose a contract
        proposing_agent = np.random.choice(list(self.agents_map.values()))
        print("Randomly selected ", proposing_agent, " to propose contract")
        contract_proposed, contract_param = proposing_agent.propose_contract()
        
        # 2. If a contract is proposed, prompt all players for voting
        if contract_proposed:
            votes = [agent.vote_on_contract(contract_param) for agent in self.agents_map.values()]
            if all(votes):
                # If all agents agree, activate the punishment function
                self.contract_active = True
        
        # 3. Prompt all agents to make actions then execute       
        for agent in self.agents_map.values():
            agent.get_orders()
            agent.execute()

        # 4. Enforce CD
        self.enforce_contract(contract_param)
        self.contract_active = False # Reset the contract flag

        # 5. Spawn new apples
        self.spawn_apples()


if __name__ == "__main__":
    world = World(15, 15)
    
    agent_1 = Agent(world, name="Alice",
                                 strategy="You want to maximize the number of apples you collect.",
                                 x = 3,
                                 y = 3,
                                 chat_model="gpt-4-0613", custom_key='openai_api_key_1_wGPT4')

    agent_2 = Agent(world, name="Bob",
                                 strategy="You want to maximize the number of apples you collect.",
                                 x = 8,
                                 y = 5,
                                 chat_model="gpt-4-0613", custom_key='openai_api_key_1_wGPT4')

    world.agents_map[agent_1.name] = agent_1
    world.agents_map[agent_2.name] = agent_2

    for i in range(5):
        print('=========== round {round} =========='.format(round=i))
        print(world)
        
        world.run(n_rounds=1)
        print(world)
        print('=========== round {round} =========='.format(round=i))
        print("\n\n\n\n\n\n\n")