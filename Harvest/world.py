import pprint
import pandas as pd
from collections import OrderedDict
from agent import *

class WorldError(Exception):
    """
    Exception raised for errors in the world. To indicate that the world is in an invalid state.
    """

    pass


class World:
    def __init__(self, x_size, y_size):
        self.x_size = x_size
        self.y_size = y_size
        self.map = [[[] for _ in range(x_size)] for _ in range(y_size)]
        self.instances = {}
        self.agents_map = OrderedDict()
        self.name_to_id = {}
        self.global_id = 0

    def is_occupied(self, x, y):
        if x < 0 or x >= self.x_size or y < 0 or y >= self.y_size:
            return True
        
        id_list = self.map[y][x]
        for id in id_list:
            if self.instances[id] != []:
                return True
        return False

    def exists(self, id):
        """
        Check if the instance exists in the world.
        """
        return id in self.instances

    def get_position(self, id):
        instance = self.instances[id]
        if instance.object_type != ObjectType.Commander:
            return instance.x, instance.y
        else:
            return None, None

    # Returns: soldier.id if a soldier is at (x, y)
    #          None       otherwise
    def get_soldier_by_position(self, x, y):
        # Out of bounds check
        if x < 0 or x >= self.x_size or y < 0 or y >= self.y_size:
            return None
        # No proxy or not a soldier
        elif self.map[y][x] != []:
            for id in self.map[y][x]:
                if self.instances[id].object_type == ObjectType.Soldier:
                    return id
        else:
            return None

    def get_object_type(self, id):
        return self.instances[id].object_type

    def get_instance_by_id(self, id):
        return self.instances[id]

    def get_instance_by_name(self, name):
        return self.instances[self.name_to_id[name.lower()]]
    
    def get_all_soldiers(self, commander_id):
        return self.commander_soldiers_map[commander_id]
    
    def get_all_instances(self):
        """return list of all instance(class) [for commander]

        Returns:
            List[SoldierProxy]: a list of instance(class)
        """
        return self.instances.values()

    def get_world_state(self):
        world_state = [[[] for _ in range(self.x_size)] for _ in range(self.y_size)]
        for instance in self.instances.values():
            if (instance.object_type == ObjectType.Soldier and instance.stamina > 0) or instance.object_type == ObjectType.Hospital:
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
        if instance.object_type == ObjectType.Commander:
            self.commander_soldiers_map[id] = []
            self.name_to_id[instance.name.lower()] = id
        elif instance.object_type == ObjectType.Soldier:
            soldiers_name_list = [
                i.name
                for i in self.instances.values()
                if i.object_type == ObjectType.Soldier
            ]
            if instance.name in soldiers_name_list:
                raise WorldError("Cannot add Soldier: name is already used")
            if self.is_occupied(instance.x, instance.y):
                raise WorldError("Cannot add Soldier: the location is occupied")
            if instance.stamina <= 0:
                raise WorldError("Cannot add Soldier: stamina is not positive")

            instance.order = "STAY"
            self.map[instance.y][instance.x].append(id)
            self.commander_soldiers_map[instance.commander_id].append(instance)
            self.name_to_id[instance.name.lower()] = id
        elif instance.object_type == ObjectType.Hospital:
            self.map[instance.y][instance.x].append(id)
            self.name_to_id[instance.name.lower()] = id
        else:
            print(instance.object_type)
            raise WorldError("Cannot add instance: invalid class are given")
        id = self.global_id
        self.instances[id] = instance
        self.global_id += 1

        return id

    def commander_update_soldiers_order(self, commander, orders):
        for soldier in self.commander_soldiers_map[commander.id]:
            action = orders[commander.get_alias_name_by_id(soldier.id)]['order']
            
            soldier.set_order(action)

    def __repr__(self):
        w, infos = self.get_world_state()
        print(pd.DataFrame.from_records(infos).sort_values("object_type").drop(
            ["id", "capacity", "occupancy", 'x', 'y'], axis=1))
        w = pd.DataFrame(w)
        idx_list = list(range(self.y_size))
        w.index = list(map(lambda s: "|" + str(s) + "|", idx_list))
        w = w.rename(columns=lambda s: "|" + str(s) + "|")
        return pprint.pformat(w, indent=4)

    def _round(self):
        # first execute all soldiers
        for comander_id, soldiers in self.commander_soldiers_map.items():
            for soldier in soldiers:
                soldier.execute()

        # then execute all public instances
        for instance in self.instances.values():
            if (
                instance.object_type != ObjectType.Commander
                and instance.object_type != ObjectType.Soldier
            ):
                instance.execute()

        # robin-round fashion
        self.commander_soldiers_map.move_to_end(next(iter(self.commander_soldiers_map.items()))[0])
        self._post_round()

    def _post_round(self):
        """
        Remove dead soldiers
        """
        for instance in list(self.instances.values()):
            if instance.object_type == ObjectType.Soldier and instance.stamina <= 0:
                self.map[instance.y][instance.x].remove(instance.id)
                self.commander_soldiers_map[instance.commander_id].remove(instance)
                del self.instances[instance.id]
                del self.name_to_id[instance.name.lower()]

    def run(self, n_rounds):
        """
        Run the world for n_rounds
        """
        for _ in range(n_rounds):
            self._round()

    def is_game_over(self):
        if len(self.commander_soldiers_map) == 1:
            return True
        else:
            return False

if __name__ == "__main__":
    world = World(10, 10)
    # Model choice: [gpt-3.5-turbo, gpt-4, gpt-4-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k-0613, text-davinci-003, gpt-4-32k, gpt-4-32k-0613]
    commander_1 = CommanderAgent(world, name="commander_1",
                                 strategy="You need to arrange all your soldier to attack the same enemy and move "
                                          "close to each other",
                                 chat_model="gpt-4-0613", custom_key='openai_api_key_1_wGPT4')
    soldier_1 = SoldierProxy(world, "Amy", 3, 4, 10, commander_1.id)
    soldier_2 = SoldierProxy(world, "Bob️", 2, 2, 10, commander_1.id)
    soldier_3 = SoldierProxy(world, "Cobb", 4, 6, 10, commander_1.id)
    soldier_4 = SoldierProxy(world, "Dave", 1, 7, 10, commander_1.id)
    soldier_5 = SoldierProxy(world, "Eva", 2, 8, 10, commander_1.id)
    
    hospital_1 = HospitalProxy(world, 0, 0, "Hospital_1")
    hospital_2 = HospitalProxy(world, 9, 9, "Hospital_2")
    hospital_3 = HospitalProxy(world, 9, 0, "Hospital_3")
    hospital_4 = HospitalProxy(world, 0, 9, "Hospital_4")


    commander_2 = CommanderAgent(world, name="commander_2",
                                 strategy="Make sure to combat each enemy with superior soldier number and try to "
                                          "encircle them to kill them as fast as possible while reduce our casualty. "
                                          "Any soldier with less than 3 stamina should retreat to the nearest "
                                          "hospital.",
                                 chat_model="gpt-4-0613", custom_key='openai_api_key_2')
    soldier_x1 = SoldierProxy(world, "Christopher", 8, 4, 10, commander_2.id)
    soldier_x2 = SoldierProxy(world, "Victoria", 8, 5, 10, commander_2.id)
    soldier_x = SoldierProxy(world, "Benjamin", 8, 6, 10, commander_2.id)
    soldier_y = SoldierProxy(world, "Alexander", 8, 7, 10, commander_2.id)
    soldier_z = SoldierProxy(world, "Theodore", 8, 8, 10, commander_2.id)

    import concurrent.futures

    def get_orders(commander):
        return commander.get_orders()


    # # world.run(n_rounds=3)
    for commander_round in range(5):
        print('=========== round {round} =========='.format(round=commander_round))
        print(world)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the tasks for parallel execution
            future1 = executor.submit(get_orders, commander_1)
            future2 = executor.submit(get_orders, commander_2)

            # Retrieve the results when they are ready
            updated_order_1 = future1.result()
            updated_order_2 = future2.result()

        print('=========== getting orders from {commander_name} =========='.format(commander_name=commander_1.name))
        # updated_order_1 = commander_1.get_orders()
        world.commander_update_soldiers_order(commander_1, updated_order_1)

        print('=========== getting orders from {commander_name} =========='.format(commander_name=commander_2.name))
        # updated_order_2 = commander_2.get_orders()
        world.commander_update_soldiers_order(commander_2, updated_order_2)
        
        world.run(n_rounds=1)
        # print(world)
        world.run(n_rounds=1)
        # print(world)
        world.run(n_rounds=1)
        print(world)
        print('=========== round {round} =========='.format(round=commander_round))
        print("\n\n\n\n\n\n\n")

    # world.run(1)
    
    # updated_order_1 = commander_1.get_orders()
    # world.commander_update_soldiers_order(commander_1.id, updated_order_1)
    # updated_order_2 = commander_2.get_orders()
    # world.commander_update_soldiers_order(commander_2.id, updated_order_2)
    
    # world.run(1)
    
    # updated_order_1 = commander_1.get_orders()
    # world.commander_update_soldiers_order(commander_1.id, updated_order_1)
    # updated_order_2 = commander_2.get_orders()
    # world.commander_update_soldiers_order(commander_2.id, updated_order_2)
    
    