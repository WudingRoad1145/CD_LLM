import sys
import datetime
from os.path import join
from pprint import pprint, pformat
import pandas as pd
from collections import OrderedDict
from agent import *
from apple import *

class WorldError(Exception):
    """
    Exception raised for errors in the world. To indicate that the world is in an invalid state.
    """
    def __init__(self, message="An error occurred in the World."):
        super().__init__(message)

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
        self.CD_memory = []
        # Randomly distribute apples
        for _ in range(num_apples):
            x, y = np.random.randint(0, x_size), np.random.randint(0, y_size)
            while self.is_occupied(x, y):
                x, y = np.random.randint(0, x_size), np.random.randint(0, y_size)
            self.add_instance(Apple(x, y))

    def is_occupied(self, x, y):
        if x < 0 or x >= self.x_size or y < 0 or y >= self.y_size:
            return True
        return any(isinstance(inst, Agent) for inst in self.map[y][x])

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
        agent_details = [instance.get_info() for instance in self.instances.values() if instance.name != "Apple"]
        return world_state, agent_details

    def add_instance(self, instance):
        id = self.global_id
        if instance.name != "Apple" and self.is_occupied(instance.x, instance.y):
            raise WorldError("Cannot add instance: the location is occupied")

        id = self.global_id
        self.instances[id] = instance
        self.map[instance.y][instance.x].append(instance)
        self.global_id += 1
        return id

    def spawn_probability(self, nearby_apples):
        #return [0, 0.005, 0.02, 0.05][min(nearby_apples, 3)]
        return [0, 0.001, 0.004, 0.01][min(nearby_apples, 3)]

    def spawn_apples(self):
        for row in range(self.x_size):
            for col in range(self.y_size):
                if not self.is_apple_at(row, col):
                    spawn_prob = self.spawn_probability(self.count_nearby_apples(row, col))
                    if np.random.random() < spawn_prob:
                        self.add_instance(Apple(row, col))

    def remove_apple(self, x, y):
        for i, apple in enumerate(self.map[y][x]):  # maybe 2 apples in 1 cell
            if isinstance(apple, Apple):
                # Find the id of the apple in self.instances
                id = next((id for id, instance in self.instances.items() if instance is apple), None)
                if id is not None:
                    del self.instances[id]
                    del self.map[y][x][i]
                    return
    
    def is_apple_at(self, x, y):
        return any(isinstance(inst, Apple) for inst in self.map[y][x])

    def count_nearby_apples(self, x, y, radius=3):
        count = 0
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                if 0 <= x+i < self.x_size and 0 <= y+j < self.y_size:
                    if any(isinstance(inst, Apple) for inst in self.map[y+j][x+i]):
                        count += 1
        return count
    
    def store_memory(self, round, contract_proposed, voting_results, exec_results, agent_rewards, contract_enforced):
        # Record historical data in memory
        self.CD_memory.append({
            'round': round,
            'contract_proposed': contract_proposed,
            'voting_results': voting_results,
            'exec_results': exec_results,
            'agent_rewards': agent_rewards,
            'contract_enforced': contract_enforced
        })
    
    def enforce_contract(self, contract_param):
        x = int(contract_param)  # The number of apples to be transferred
        
        # Check each agent to see if they violated the contract
        for agent_id, agent in self.agents_map.items():
            if agent.just_collected_apple: 
                nearby_apples = self.count_nearby_apples(agent.x, agent.y)
                print("enforcing contract on agent", agent.name)
                
                # If the agent collected an apple in a low-density region
                if nearby_apples < 4:
                    # Take x apples from the violating agent
                    agent.rewards = max(0, agent.rewards - x)
                    print(agent.name, "'s reward minus", x)
                    
                    # Distribute the apples equally among the other agents
                    num_other_agents = len(self.agents_map) - 1
                    apples_per_agent = x // num_other_agents
                    for other_agent_id, other_agent in self.agents_map.items():
                        if other_agent_id != agent_id:
                            other_agent.rewards += apples_per_agent
                    return "Agent " + agent.name + " violated the contract and thus distributed "+ x + " apples to other agents."
        
        return "Nobody violated the contract."


    def __repr__(self):
        w, infos = self.get_world_state()
        print(pd.DataFrame.from_records(infos).sort_values("name").drop(
            ["id"], axis=1))
        w = pd.DataFrame(w)
        idx_list = list(range(self.y_size))
        w.index = list(map(lambda s: "|" + str(s) + "|", idx_list))
        w = w.rename(columns=lambda s: "|" + str(s) + "|")
        return pformat(w, indent=4)

    def run(self, n_rounds, contract_template):
        """
        Run the world for n_rounds
        """
        for _ in range(n_rounds):
            # Initialize agent template
            intro = "You are a player in a 2D grid-based world who can move around to collect apples. {strategy} There are {n_agents} players in total. Everyone wants to collect as many apples as possible. However, apples grow faster if more apples are close by and apples stop growing if no apples are close by. We would run {total_round} rounds. This is round {current_round}."
            [agent.message_history.append(intro.format(
                strategy=agent.strategy, 
                n_agents = len(self.agent_map),
                total_round=n_rounds,
                current_round=_)) 
             for agent in self.agents_map.values()]

            print('=========== round {round} =========='.format(round=i))
            print(world)
            print("**************************************************************************")
            self._round(contract_template)
            print(world)
            print('=========== round {round} =========='.format(round=i))
            print("\n\n\n\n\n\n\n")


    def _round(self, round_number, contract_template):
        # 1. Randomly pick one agent to propose a contract
        proposing_agent = np.random.choice(list(self.agents_map.values()))
        print("Randomly selected", proposing_agent.name, "to propose contract")
        contract_proposed, contract_param = proposing_agent.propose_contract(contract_template)
        
        # 2. If a contract is proposed, prompt all players for voting
        if contract_proposed:
            # Exclude the proposing agent from the voting process
            voting_results = [(agent.name, agent.vote_on_contract(proposing_agent.name, contract_template, contract_param))
                    for agent in self.agents_map.values() if agent.name != proposing_agent.name]
            print(voting_results)
            if all(vote for _, vote in voting_results): 
                # If all agents agree, activate the punishment function
                self.contract_active = True
        
        # 3. Prompt all agents to make actions then execute
        exec_results = []       
        for agent in self.agents_map.values():
            action = agent.get_action(contract_template, contract_param)
            final_action = agent.execute(action)
            exec_results.append(final_action)

        # 4. Enforce CD
        enforcement_result = self.enforce_contract(contract_param)
        self.contract_active = False # Reset the contract flag

        # 5. Record CD history
        final_contract = contract_template.replace("X", contract_param)
        self.store_memory(round_number, final_contract, voting_results, exec_results, {agent.name: agent.rewards for agent in self.agents_map.values()}, enforcement_result)

        # 6. Spawn new apples
        self.spawn_apples()


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join("Harvest/logs", f"output_{timestamp}.txt")
    sys.stdout = open(filename, 'w')

    world = World(20, 20, 20) # 15x15 world with 20 apples
    
    agent_1 = Agent(world, name="Alice",
                                 strategy="You want to collect as many apples as possible. You want to help others collect more apples as well so that the society gets better off.",
                                 x = 3,
                                 y = 3,
                                 chat_model="gpt-4-0613", custom_key='openai_api_key_1_wGPT4')

    agent_2 = Agent(world, name="Bob",
                                 strategy="You want to maximize the number of apples you collect. You don't want to overconsume apples because you want to sustainably harvest apples.",
                                 x = 12,
                                 y = 16,
                                 chat_model="gpt-4-0613", custom_key='openai_api_key_1_wGPT4')
    agent_3 = Agent(world, name="Cao",
                                 strategy="You want to maximize the number of apples you collect. You would love collaborate with others to collect more apples in multiple rounds, not just in one round.",
                                 x = 9,
                                 y = 7,
                                 chat_model="gpt-4-0613", custom_key='openai_api_key_1_wGPT4')

    world.agents_map[agent_1.name] = agent_1
    world.agents_map[agent_2.name] = agent_2
    world.agents_map[agent_3.name] = agent_3

    contract_template = "When an agent takes a consumption action of an apple in a low-density region, defined as an apple having less than 4 neighboring apples within a radius of 5, they transfer X apples to the other agents, which is equally distributed to the other agents."

    world.run(n_rounds=10,contract_template=contract_template)

    sys.stdout.close()