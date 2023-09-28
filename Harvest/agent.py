import os
import json
from pprint import pprint
from langchain.schema import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from world import World, WorldError
import numpy as np

# basic moves every agent should do
BASE_ACTIONS = {
    0: "MOVE_LEFT",  # Move left
    1: "MOVE_RIGHT",  # Move right
    2: "MOVE_UP",  # Move up
    3: "MOVE_DOWN",  # Move down
    4: "STAY",  # don't move
    5: "COLLECT",  # collect apple
}

class AgentError(Exception):
    """
    Exception raised for errors in commander. To feed back into the chatbot.
    """

    pass


class Agent():
    def __init__(self, world: World, name: str, strategy: str, chat_model: str = 'gpt-4',
                 max_retry_times: int = 5, custom_key: str = None, custom_key_path: str = 'api_key/llm_api_keys.json'):
        """
        This function initializes the LLM agent.
        Options for chat_model keyword:[gpt-3.5-turbo, gpt-4, gpt-4-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k-0613, text-davinci-003, gpt-4-32k, gpt-4-32k-0613]
        """
        self.name = name
        self.strategy = strategy # high level strategy for the player. e.g. you want to maximize your own benefits but not harm others
        self.world = world
        self.points = 0 # initialize # apples collected
        # get json API key from api_key/llm_api_key.json
        if custom_key:
            with open('api_key/llm_api_keys.json', 'r') as file:
                api_keys = json.load(file)

        if 'gpt' in chat_model:
            print("Loading GPT chat model...")  # log in the future
            self.chat_model = ChatOpenAI(
                model_name=chat_model,
                openai_api_key= api_keys[custom_key] if custom_key else os.environ.get("OPENAI_API_KEY"),
                temperature=0, max_tokens=1500, request_timeout=120, verbose=True)
        elif "claude" in chat_model:
            print("Loading Claude chat model...")  # log in the future
            self.chat_model = ChatAnthropic(
                model=chat_model,
                anthropic_api_key=api_keys[custom_key] if custom_key else os.environ.get("ANTHROPIC_API_KEY"),
                temperature=0, verbose=True)
        else:
            raise NotImplementedError(
                f"Chat model {chat_model} not implemented.")

        self.message_history = []
        self.max_retry_times = max_retry_times
        self.remaining_retry_times = max_retry_times

        self.id = world.add_instance(self)

    def reset(self):
        self.message_history = []
        self.remaining_retry_times = self.max_retry_times

    def collect_apple(self, x, y):
        """
        Logic for the agent to collect an apple at position (x, y).
        """
        if self.world.is_apple_at(x, y): #TODO
            self.world.remove_apple(x, y) #TODO
            self.points += 1
            
    def execute(self):
        action, error_message = self.take_action()
        if error_message is not None:
            #TODO: return error message not raise
            # raise world.WorldError(error_message)
            # TODO: give this to commander
            return
        if "MOVE" in action:
            _, dir = action.split(" ")
            self._move(dir)
        elif "ATTACK" in action:
            _, target_id = action.split(" ")
            self._attack(int(target_id))
        elif "STAY" in action:
            self._stay()

    def get_input_prompt(self) -> str:
        all_instances = self.world.get_all_instances()
        n_agents = len([s for s in all_instances if s.object_type ==
                         ObjectType.Agent and s.commander_id == self.id and s.stamina > 0])
        current_game_state = [
            [[] for _ in range(self.world.x_size)] for _ in range(self.world.y_size)]
        for instance in all_instances:
            match instance.object_type:
                case ObjectType.Soldier:
                    # check if soldier is my soldier or enemy's
                    if instance.commander_id == self.id:
                        # then add its information onto the current_game_state in this format:
                        # soldier_5 {"stamina": 2, "current_order": "GO enemy_3"}
                        current_game_state[instance.y][instance.x].append(
                            f"soldier_{instance.name} {{\"stamina\": {instance.stamina}, \"current_order\": \"{instance.order}\"}}")
                    else:
                        # then add its information onto the current_game_state in this format:
                        # enemy_5 {"stamina": 2, "current_order": "GO enemy_3"}
                        current_game_state[instance.y][instance.x].append(
                            f"enemy_{instance.name} {{\"stamina\": {instance.stamina}, \"current_order\": \"{instance.order}\"}}")
                case ObjectType.Hospital:
                    # add the hospital information onto the current_game_state in this format: hospital {"state": full / have extra room}
                    hospital_state = "full" if instance.capacity == instance.occupancy else "have extra room"
                    current_game_state[instance.y][instance.x].append(
                        f"{instance.name} {{\"state\": {hospital_state}}}")

        current_game_state = [
            [" and ".join(i) if i != [] else "" for i in j] for j in current_game_state]
        current_game_state = "\n".join(
            ["|".join(row) for row in current_game_state])

        prompt_input = """
You are a player of {n_agents} players in a 2D grid-based world. You are perfectly rational and you want to maximize the number of apples you have.

You can choose one of the following actions:
- GO [up/down/left/right/target]: you will move in the following direction for 1 grid.
- STAY: soldier will not move and stay at the original location.
- Collect: Collect the apple in the current grid.

For example:
"GO down": you will move down the map for 1 grid.
"STAY": you will just stay at the same location doing nothing.
"COLLECT": you will collect the apple in the current grid.

This is the current game state:

{current_game_state}

---

You have the option of proposing a contract to the other players to prevent overconsume apples. If the contract is agreed by the majority, it will be enforced. If you want to propose such a contract, please follow the format {contract} and decide the variable X. If you don't want to propose such a contract, simply 

---

In the reply, please reason step by step. You should give your order in the following format as shown in the order section of the previous json. Please update order that you think is wrong with reasoning.
All your output should be aggregated into a json file with the following format:
```json
{{
    "general thoughts": "TODO",
    "reasoning": "TODO",
    "contract": "TODO",
    "order": "TODO"
}}
```
        """.format(n_agents=n_agents, current_game_state=current_game_state, contract=self.contract)
        
        self.message_history.append(HumanMessage(content=prompt_input))
        
        # return variables we need when check the output meets the requirement
        return n_agents

    def get_orders(self, verbose_input=False, verbose_output=True) -> dict:
        """
        This function takes in the input_prompt and returns a dictionary of order.
        """

        expect_soldier_number = self.get_input_prompt()
        if verbose_input:
            print(f"input_prompt: {self.message_history}")
        _output = self.chat_model(self.message_history)
        json_string = _output.content.split(
            "```json")[-1].strip().replace('```', '')
        try:
            output = json.loads(json_string)
            self.message_history.append(AIMessage(content=json_string))
            error_execution_message = self.get_error_execution_message(output)
            if error_execution_message != "":
                raise CommanderError(error_execution_message)
            if expect_soldier_number != len(list(output.keys())) - 1:
                raise CommanderError(
                    f"You are missing some soldiers in your output. Please make sure you have orders for all {expect_soldier_number} soldiers.")
        except Exception as e:
            if e.__class__.__name__ == "CommanderError":
                print(e)
            else:
                print(f"output is not in json format: {json_string}")

            if self.remaining_retry_times > 0:
                if e.__class__.__name__ == "CommanderError":
                    error_message = HumanMessage(
                        content=e.__str__())
                else:
                    error_message = HumanMessage(
                        content=f"Your output is not in json format. Please make sure your output is in the following format:\n```json\n{{\n    \"general thoughts\": \"TODO\",\n    \"soldier_1\": {{\n      \"reasoning\": \"TODO\",\n      \"order\": \"TODO\"\n    }},\n...\n    \"soldier_{expect_soldier_number}\": {{\n      \"reasoning\": \"TODO\",\n      \"order\": \"TODO\"\n    }}\n}}\n```")
                self.message_history.append(error_message)
                self.remaining_retry_times -= 1

                return self.get_orders()
            else:
                raise CommanderError(
                    "You have exceeded the maximum number of retries. Please try again later.")

        if verbose_output:
            pprint(output)
        self.reset()
        return output

    def get_info(self):
        return {"id": self.id, "name": self.name, "object_type": self.object_type}

    def get_error_execution_message(self, output):
        error_message = []
        soldiers_id_list = self.world.get_all_soldiers()
        for id in soldiers_id_list:
            instance = self.world.get_instance_by_id(id)
            order = output[self.get_alias_name_by_id(id)]["order"]
            
            # check if the order will cause the soldier to move out of the map
            order = instance.unalias_order(order)
            if order.startswith("GO"):
                target = order.split(" ")[1].lower()
                if target in ["up", "down", "left", "right"]:
                    dir = target
                    x = instance.x + instance.directions[dir.upper()][0]
                    y = instance.y + instance.directions[dir.upper()][1]
                    if x < 0 or x >= self.world.x_size or y < 0 or y >= self.world.y_size:
                        error_message.append(f"Your soldier {instance.name} is trying to move out of the map, which is not allowed. ")
                if target.startswith("enemy") or target.startswith("soldier"):
                    target_name = target.split("_", 1)[1]
                    if target_name.lower() not in self.world.name_to_id:
                        error_message.append(f"Your soldier {instance.name} is trying to move to a non-existing target, which is not allowed.")
        
        error_message.append(HumanMessage(content="Generate new reply based on information above."))
        
        return "\n".join(error_message)
            

if __name__ == "__main__":
    pass

