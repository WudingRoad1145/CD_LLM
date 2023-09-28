import pprint
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
    def __init__(self, x_size, y_size, num_apples=10):
        self.x_size = x_size
        self.y_size = y_size
        self.map = [[[] for _ in range(x_size)] for _ in range(y_size)]
        self.instances = {}
        self.agents_map = OrderedDict()
        self.name_to_id = {}
        self.global_id = 0
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
        SPAWN_PROB = [0, 0.005, 0.02, 0.05]
        APPLE_RADIUS = 2

        new_apple_points = []
        for apple in [i for i in self.instances.values() if i.object_type == "Apple"]:
            row, col = apple.x, apple.y
            num_apples = 0
            for j in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                for k in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                    if j ** 2 + k ** 2 <= APPLE_RADIUS ** 2:
                        x, y = row + j, col + k
                        if 0 <= x < self.x_size and 0 <= y < self.y_size:
                            if any(isinstance(inst, Apple) for inst in self.map[y][x]):
                                num_apples += 1

            spawn_prob = SPAWN_PROB[min(num_apples, 3)]
            if np.random.random() < spawn_prob:
                new_x, new_y = row + np.random.randint(-1, 2), col + np.random.randint(-1, 2)
                if 0 <= new_x < self.x_size and 0 <= new_y < self.y_size and not self.is_occupied(new_x, new_y):
                    new_apple_points.append((new_x, new_y))

        for x, y in new_apple_points:
            self.add_instance(Apple(x, y))

    def consume_apple(self, agent_id, x, y):
        # Add logic for agent with agent_id to consume apple at (x, y)
        # TODO
        pass

    def agent_update_order(self, agent, order):
        #TODO
        pass

    def __repr__(self):
        w, infos = self.get_world_state()
        print(pd.DataFrame.from_records(infos).sort_values("name").drop(
            ["id", "capacity", "occupancy", 'x', 'y'], axis=1))
        w = pd.DataFrame(w)
        idx_list = list(range(self.y_size))
        w.index = list(map(lambda s: "|" + str(s) + "|", idx_list))
        w = w.rename(columns=lambda s: "|" + str(s) + "|")
        return pprint.pformat(w, indent=4)

    def run(self, n_rounds):
        """
        Run the world for n_rounds
        """
        for _ in range(n_rounds):
            self._round()

    def _round(self):
        # Execute all agents
        for agent in self.instances.values():
            agent.execute()

        # Execute all public instances
        for instance in self.instances.values():
            instance.execute()

        # Spwan new apples
        self.spawn_apples()


if __name__ == "__main__":
    world = World(15, 15)
    
    agent_1 = Agent(world, name="Alice",
                                 strategy="You want to maximize the number of apples you collect.",
                                 chat_model="gpt-4-0613", custom_key='openai_api_key_1_wGPT4')

    agent_2 = Agent(world, name="Bob",
                                 strategy="You want to maximize the number of apples you collect.",
                                 chat_model="gpt-4-0613", custom_key='openai_api_key_2')

    for i in range(5):
        print('=========== round {round} =========='.format(round=i))
        print(world)

        updated_order_1 = agent_1.get_orders()
        world.agent_update_order(agent_1, updated_order_1)

        updated_order_2 = agent_2.get_orders()
        world.agent_update_order(agent_2, updated_order_2)
        
        world.run(n_rounds=1)
        world.run(n_rounds=1)
        world.run(n_rounds=1)
        print(world)
        print('=========== round {round} =========='.format(round=i))
        print("\n\n\n\n\n\n\n")