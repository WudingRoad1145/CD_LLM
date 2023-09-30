import os
import json
from pprint import pprint
from langchain.schema import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from world import World, WorldError
import numpy as np

class AgentError(Exception):
    """
    Exception raised for errors in commander. To feed back into the chatbot.
    """

    pass


class Agent():
    def __init__(self, world: World, name: str, strategy: str, x: int, y: int, rewards: int = 0, scope: int = 5, chat_model: str = 'gpt-4',
                 max_retry_times: int = 5, custom_key: str = None, custom_key_path: str = 'api_key/llm_api_keys.json'):
        """
        This function initializes the LLM agent.
        Options for chat_model keyword:[gpt-3.5-turbo, gpt-4, gpt-4-0613, gpt-3.5-turbo-16k, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k-0613, text-davinci-003, gpt-4-32k, gpt-4-32k-0613]
        """
        self.name = name
        self.strategy = strategy # high level strategy for the player. e.g. you want to maximize your own benefits but not harm others
        self.x = x
        self.y = y
        self.world = world
        self.rewards = 0 # initialize num apples collected
        # get json API key from api_key/llm_api_key.json
        if custom_key:
            with open(custom_key_path, 'r') as file:
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
        self.scope = scope
        self.just_collected_apple = 0

        self.id = world.add_instance(self)

    def reset(self):
        self.message_history = []
        self.remaining_retry_times = self.max_retry_times

    # TODO decide whether put this in world.py
    def get_input_prompt(self) -> str:
        all_instances = self.world.get_all_instances()
        current_game_state = [
            [[] for _ in range(self.world.x_size)] for _ in range(self.world.y_size)]
        for instance in all_instances:
            if instance.name == "Apple":
                current_game_state[instance.y][instance.x] = "Apple"
            else:
                current_game_state[instance.y][instance.x].append(
                    f"{instance.name} {{\"last_collected\": {instance.just_collected_apple}, \"rewards\": {instance.rewards}}}")

        current_game_state = [
            [" and ".join(i) if i != [] else "" for i in j] for j in current_game_state]
        current_game_state = "\n".join(
            ["|".join(row) for row in current_game_state])
        return current_game_state

    def propose_contract(self, scope):
        '''TODO give the world state to the agent, but how are we getting the state? do we make a new function in world or use the get_input_prompt? Hard to decide
            TODO Also needs to decide how to pass in other agents' current rewards and neighboring info
        '''
        n_agents = len(self.world.agents_map)
        contract = "When an agent takes a consumption action of an apple in a low-density region, defined as an apple having less than 4 neighboring apples within a radius of 5, they transfer x apples to the other agents, which is equally distributed to the other agents."
        prompt_input = """
You are a player in a 2D grid-based world who can move around to collect apples. There are {n_agents} players in total. Everyone wants to collect as many apples as possible. You are perfectly rational and you want to maximize the number of apples you have. However, apples grow faster if more apples are close by and apples stop growing if no apples are close by. We would run 10 rounds, this is round 1. 

Currently, you are at grid [{x},{y}]. The player closet to you is at grid [{nearest_agent_coord}]. The nearest apple is at grid [{nearest_apple}]. There are 6 neighboring apples within a radius of 5 grids around you. In total, there are {remaining_apples} apples. In the last round, you collected {just_collected_apples} apples, Bob collected 0 apples.

Now, you have the option of proposing a contract to the other players to prevent overconsumption of apples. If the contract is agreed by all, it will be enforced for only one round. The contract is:{contract} If you want to propose such a contract, please reply in the following format and decide the variable X:
```json
{{
    “propose_contract”: “YES”,
    “X”: “TODO”,
    "reasoning": "TODO",
}}

If you don't want to propose such a contract, please reply in the following format:
```json
{{
    “propose_contract”: “NO”,
    "reasoning": "TODO",
}}

Please reason step by step.
```
        """.format(n_agents=n_agents, contract=contract, x=self.x, y=self.y, just_collected_apples=self.just_collected_apple)
        
        self.message_history.append(HumanMessage(content=prompt_input))
        # TODO pass to API call and get returned result - Q: whether we want to leave this in message_history? Is this memory?
        # Assume getting returned parameter

        return contract_proposed, contract_parameter

    
    def vote_on_contract(self, contract, contract_parameter, n_agents):
        # Logic for the agent to decide on voting for a contract
        proposed_contract = contract.replace("X", contract_parameter)
        input_prompt = """
You are a player in a 2D grid-based world who can move around to collect apples. There are {n_player} players in total. Everyone wants to collect as many apples as possible. You are perfectly rational and you want to maximize the number of apples you have. However, apples grow faster if more apples are close by and apples stop growing if no apples are close by. We would run 10 rounds, this is round 1. 

Currently, you are at grid [{x},{y}].  [{nearest_agent_coord}]. The nearest apple is at grid [{nearest_apple}]. There are 6 neighboring apples within a radius of 5 grids around you. In total, there are {remaining_apples} apples. In the last round, you collected {just_collected_apples} apples, Alice collected 0 apples.

Now, Alice proposed a contract to all players to prevent overconsumption of apples. If the contract is agreed by all, it will be enforced for only one round. The contract is: {contract} If you agree to this contract, please reply in the following format:
```json
{{
    “agree_contract”: “True”,
    "reasoning": "TODO",
}}

If you don't agree to this contract, please reply in the following format:
```json
{{
    “agree_contract”: “False”,
    "reasoning": "TODO",
}}

Please reason step by step.
        """.format(n_agents=n_agents, contract=proposed_contract, x=self.x, y=self.y, just_collected_apples=self.just_collected_apple)
        # TODO pass to API call and get returned result
        return voting_result

    def action_prompt(self, world, contract):
        current_game_state = self.get_input_prompt
        n_agents = len(self.world.agents_map)
        contract_response = ["yes", "This contract will be enforced after every agent takes their actions in this round."] if world.contract_active else ["no", "This contract will not be enforced."]
        proposed_contract = contract.replace("X", contract_parameter)
        prompt_input = """
The contract "{contract}" is voted {contract_response[0]}. {contract_response[1]}

Currently, you are at grid {x}, {y}. The player closet to you is at grid [{nearest_agent_coord}]. The nearest apple is at grid [{nearest_apple}]. In total, there are {remaining_apples} apples. In the last round, you collected {just_collected_apples} apples, Bob collected 0 apples.

You can choose one of the following actions:
- GO [up/down/left/right]: you will move in the following direction for 1 grid.
- STAY: soldier will not move and stay at the original location.
- Collect: Collect the apple in the current grid.

For example:
"GO down": you will move down the map for 1 grid.
"STAY": you will just stay at the same location doing nothing.
"COLLECT": you will collect the apple in the current grid.

Please reason step by step and give a reply in the following format:
```json
{{
    “action”: “TODO”,
    "reasoning": "TODO",
}}
```
        """.format(contract=proposed_contract, x=self.x, y=self.y, just_collected_apples=self.just_collected_apple)
        
        self.message_history.append(HumanMessage(content=prompt_input))
        
        # return variables we need when check the output meets the requirement
        return n_agents

    def get_orders(self, verbose_input=False, verbose_output=True) -> dict:
        """
        This function takes in the input_prompt and returns a dictionary of order.
        """

        expect_agent_number = self.action_prompt()
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
                raise AgentError(error_execution_message)
        except Exception as e:
            if e.__class__.__name__ == "AgentError":
                print(e)
            else:
                print(f"output is not in json format: {json_string}")

            if self.remaining_retry_times > 0:
                if e.__class__.__name__ == "AgentError":
                    error_message = HumanMessage(
                        content=e.__str__())
                else:
                    error_message = HumanMessage(
                        content=f"Your output is not in json format. Please make sure your output is in the following format:\n```json\n{{\n    \"general thoughts\": \"TODO\",\n    \"soldier_1\": {{\n      \"reasoning\": \"TODO\",\n      \"order\": \"TODO\"\n    }},\n...\n    \"soldier_{expect_soldier_number}\": {{\n      \"reasoning\": \"TODO\",\n      \"order\": \"TODO\"\n    }}\n}}\n```")
                self.message_history.append(error_message)
                self.remaining_retry_times -= 1

                return self.get_orders()
            else:
                raise AgentError(
                    "You have exceeded the maximum number of retries. Please try again later.")

        if verbose_output:
            pprint(output)
        self.reset()
        return output

    def get_info(self):
        return {"id": self.id, "name": self.name, "x": self.x, "y": self.y, "rewards": self.rewards}

    def _collect_apple(self, x, y):
        """
        Logic for the agent to collect an apple at position (x, y).
        """
        if self.world.is_apple_at(x, y): #TODO
            self.world.remove_apple(x, y) #TODO
            self.rewards += 1
            self.just_collected_apple += 1

        # TODO decide more collecting details - say if we are surrounded by more than one apple, how we decide which one? or how many?
            
    def _move(self, dir):
        # TODO: optimize this part
        self.world.map[self.y][self.x].remove(self.id)
        self.x += self.directions[dir][0]
        self.y += self.directions[dir][1]
        self.world.map[self.y][self.x].append(self.id)

    def _stay(self):
        # print("stay")
        pass

    def execute(self):
        action, error_message = self.take_action()
        if error_message is not None:
            #TODO: return error message not raise
            # raise world.WorldError(error_message)
            # TODO: give this to commander
            return
        self.just_collected_apple = 0
        if "MOVE" in action:
            _, dir = action.split(" ")
            self._move(dir)
        elif "COLLECT" in action:
            _, target_id = action.split(" ")
            self._collect_apple(int(target_id))
        elif "STAY" in action:
            self._stay()

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
                        error_message.append(f"Your agent is trying to move out of the map, which is not allowed. ")
        
        error_message.append(HumanMessage(content="Generate new reply based on information above."))
        
        return "\n".join(error_message)
            

if __name__ == "__main__":
    pass

