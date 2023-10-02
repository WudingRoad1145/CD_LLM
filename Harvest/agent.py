import os
import json
from pprint import pprint
from langchain.schema import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from world import World, WorldError
import pandas as pd
import numpy as np
from utils import *

class AgentError(Exception):
    """
    Exception raised for errors in agent prompting. To feed back into the chatbot.
    """
    def __init__(self, message="An error occurred in the Agent."):
        super().__init__(message)

class Agent:
    def __init__(self, world: World, name: str, strategy: str, x: int, y: int, rewards: int = 0,
                 chat_model: str = 'gpt-4', max_retry_times: int = 5,
                 custom_key: str = None, custom_key_path: str = 'api_key/llm_api_keys.json'):
        self.name = name
        self.strategy = strategy
        self.x = x
        self.y = y
        self.world = world
        self.rewards = 0
        self.directions = {
            "UP": [0, -1],
            "DOWN": [0, 1],
            "LEFT": [-1, 0],
            "RIGHT": [1, 0]
        }
        custom_key_path = os.path.join('api_key', 'llm_api_keys.json')
        api_keys = self._get_api_keys(custom_key, custom_key_path)
        self.chat_model = self._init_chat_model(chat_model, api_keys, custom_key)
        self.message_history = []
        self.max_retry_times = max_retry_times
        self.remaining_retry_times = max_retry_times
        self.just_collected_apple = 0
        self.id = world.add_instance(self)

    def _get_api_keys(self, custom_key, custom_key_path):
            if custom_key:
                with open(custom_key_path, 'r') as file:
                    return json.load(file)
            return None

    def _init_chat_model(self, chat_model, api_keys, custom_key):
        print(f"Loading {chat_model.split('-')[0].upper()} chat model...")
        if 'gpt' in chat_model:
            return ChatOpenAI(
                model_name=chat_model,
                openai_api_key=api_keys[custom_key] if custom_key else os.environ.get("OPENAI_API_KEY"),
                temperature=0, max_tokens=1500, request_timeout=120, verbose=True)
        elif "claude" in chat_model:
            return ChatAnthropic(
                model=chat_model,
                anthropic_api_key=api_keys[custom_key] if custom_key else os.environ.get("ANTHROPIC_API_KEY"),
                temperature=0, verbose=True)
        else:
            raise NotImplementedError(f"Chat model {chat_model} not implemented.")

    def get_info(self):
        return {"id": self.id, "name": self.name, "x_coord": self.x, "y_coord": self.y, "total_rewards": self.rewards, "just_collected_apple": self.just_collected_apple}


    def reset(self):
        self.message_history = []
        self.remaining_retry_times = self.max_retry_times

    def nearest_apple(self, agent_x, agent_y, world_state):
        # If there's an apple on the agent's current position, return it as the nearest
        if 'Apple' in world_state[agent_y][agent_x]:
            return (agent_x, agent_y)
        
        apple_distances = [(i, j, distance(agent_x, agent_y, j, i)) for i, row in enumerate(world_state) for j, cell in enumerate(row) if 'Apple' in cell]
        nearest_apple_y, nearest_apple_x, _ = min(apple_distances, key=lambda x: x[2])
        return (nearest_apple_x, nearest_apple_y)

    def count_remaining_apple(self, world_state):
        apple_count = 0
        for row in world_state:
            for cell in row:
                apple_count += cell.strip().split(' & ').count('Apple')
        return apple_count

    def propose_contract(self, contract, scope=3):
        ''' 
            Logic for a randomly selected agent to propose a contract with parameter

            return: contract_proposed:bool, contract_parameter:str
        '''
        world_state, agent_details = self.world.get_world_state()
        # Convert agent details to DataFrame for easier processing
        agent_df = pd.DataFrame(agent_details)
        other_agents = agent_df[agent_df['name'] != self.name]
        # Just collected apples prompting
        agent_sentences = ["{} collected {} apples".format(row['name'], row['just_collected_apple']) for _, row in other_agents.iterrows()]
        collected_apples_sentence = "In the last round, you collected {} apples, {}.".format(self.just_collected_apple, ", ".join(agent_sentences))
        
        # Calculate the nearest agent's coordinates
        agent_df['distance'] = agent_df.apply(lambda row: distance(self.x, self.y, row['x_coord'], row['y_coord']) if row['name'] != self.name else np.inf, axis=1)
        nearest_agent = agent_df.loc[agent_df['distance'].idxmin()]
        nearest_agent_coord = (nearest_agent['x_coord'], nearest_agent['y_coord'])

        # Calculate the nearest apple's coordinates
        nearest_apple_coord = self.nearest_apple(self.x, self.y, world_state)

        # Calculate the number of remaining apples
        remaining_apples = self.count_remaining_apple(world_state)

        # Calculate the number of neighboring apples 
        neighbor_apple = self.world.count_nearby_apples(self.x,self.y,scope)  

        memory_sentence = ", ".join([f"In round {i}, contract proposed was {mem['contract_proposed']}, voting results were {mem['voting_results']}, agent rewards were {mem['agent_rewards']}, and contract enforced was {mem['contract_enforced']}." for i, mem in enumerate(self.world.CD_memory, 1)]) if self.world.CD_memory != [] else ""

        input_prompt = """
You are a player in a 2D grid-based world who can move around to collect apples. {strategy} There are {n_agents} players in total. Everyone wants to collect as many apples as possible. You are perfectly rational and you want to maximize the number of apples you have. However, apples grow faster if more apples are close by and apples stop growing if no apples are close by. We would run multiple rounds. 

Currently, you are at grid ({x},{y}). The player closet to you is at grid {nearest_agent_coord}. The nearest apple is at grid {nearest_apple_coord}. There are {neighbor_apple} neighboring apples within a radius of {scope} grids around you. In total, there are {remaining_apples} apples. {collected_apples_sentence}

Here is the world state in your scope:\n
{world_state}

{CD_memory}

Now, you have the option of proposing a contract to the other players to prevent overconsumption of apples. If the contract is agreed by all, it will be enforced for only one round. The contract is:{contract} If you want to propose such a contract, please reply in the following format and decide the variable X:
```json
{{
    “propose_contract”: “TRUE”,
    “X”: “TODO”,
    "reasoning": "TODO",
}}

If you don't want to propose such a contract, please reply in the following format:
```json
{{
    “propose_contract”: “FALSE”,
    "reasoning": "TODO",
}}

Please reason step by step.
```
        """.format(
        strategy=self.strategy,
        n_agents=len(agent_details),
        contract=contract,
        x=self.x,
        y=self.y,
        just_collected_apples=self.just_collected_apple,
        world_state="\n".join([" | ".join(row) for row in world_state]),
        nearest_agent_coord=nearest_agent_coord,
        nearest_apple_coord=nearest_apple_coord,
        scope=scope,
        remaining_apples=remaining_apples,
        neighbor_apple=neighbor_apple,
        collected_apples_sentence=collected_apples_sentence,
        CD_memory=memory_sentence,
    )
        #print(prompt_input)
        self.message_history.append(HumanMessage(content=input_prompt))
        output = self.call_LLM()
        if output['propose_contract'] == 'TRUE':
            contract_proposed = True
            contract_parameter = output['X']
        else:
            contract_proposed = False
            contract_parameter = {}

        return contract_proposed, contract_parameter

    
    def vote_on_contract(self, proposer_name, contract, contract_parameter, scope=3):
        '''
            Logic for the agent to decide on voting for a contract

            Return: voting_result:bool
        '''
        world_state, agent_details = self.world.get_world_state()
        # Convert agent details to DataFrame for easier processing
        agent_df = pd.DataFrame(agent_details)
        other_agents = agent_df[agent_df['name'] != self.name]
        # Just collected apples prompting
        agent_sentences = ["{} collected {} apples".format(row['name'], row['just_collected_apple']) for _, row in other_agents.iterrows()]
        collected_apples_sentence = "In the last round, you collected {} apples, {}.".format(self.just_collected_apple, ", ".join(agent_sentences))
        
        # Calculate the nearest agent's coordinates
        agent_df['distance'] = agent_df.apply(lambda row: distance(self.x, self.y, row['x_coord'], row['y_coord']) if row['name'] != self.name else np.inf, axis=1)
        nearest_agent = agent_df.loc[agent_df['distance'].idxmin()]
        nearest_agent_coord = (nearest_agent['x_coord'], nearest_agent['y_coord'])

        # Calculate the nearest apple's coordinates
        nearest_apple_coord = self.nearest_apple(self.x, self.y, world_state)

        # Calculate the number of remaining apples
        remaining_apples = self.count_remaining_apple(world_state)

        # Calculate the number of neighboring apples 
        neighbor_apple = self.world.count_nearby_apples(self.x,self.y,scope)  

        final_contract = contract.replace("X", contract_parameter)

        memory_sentence = ", ".join([f"In round {i}, contract proposed was {mem['contract_proposed']}, voting results were {mem['voting_results']}, agent rewards were {mem['agent_rewards']}, and contract enforced was {mem['contract_enforced']}." for i, mem in enumerate(self.world.CD_memory, 1)]) if self.world.CD_memory != [] else ""

        input_prompt = """
You are a player in a 2D grid-based world who can move around to collect apples. There are {n_agents} players in total. Everyone wants to collect as many apples as possible. You are perfectly rational and you want to maximize the number of apples you have. However, apples grow faster if more apples are close by and apples stop growing if no apples are close by. We would run multiple rounds. 

Currently, you are at grid ({x},{y}). The player closet to you is at grid {nearest_agent_coord}. The nearest apple is at grid {nearest_apple_coord}. There are {neighbor_apple} neighboring apples within a radius of {scope} grids around you. In total, there are {remaining_apples} apples. {collected_apples_sentence}

Here is the world state in your scope:\n
{world_state}

{CD_memory}

Now, {proposer} proposed a contract to all players to prevent overconsumption of apples. If the contract is agreed by all, it will be enforced for only one round. The contract is: {contract} If you agree to this contract, please reply in the following format:
```json
{{
    “agree_contract”: “TRUE”,
    "reasoning": "TODO",
}}

If you don't agree to this contract, please reply in the following format:
```json
{{
    “agree_contract”: “FALSE”,
    "reasoning": "TODO",
}}

Please reason step by step.
        """.format(
        n_agents=len(agent_details),
        x=self.x,
        y=self.y,
        just_collected_apples=self.just_collected_apple,
        world_state="\n".join([" | ".join(row) for row in world_state]),
        nearest_agent_coord=nearest_agent_coord,
        nearest_apple_coord=nearest_apple_coord,
        scope=scope,
        remaining_apples=remaining_apples,
        neighbor_apple=neighbor_apple,
        collected_apples_sentence=collected_apples_sentence,
        contract=final_contract,
        proposer=proposer_name,
        CD_memory=memory_sentence,
    )
        self.message_history.append(HumanMessage(content=input_prompt))
        output = self.call_LLM()
        if output['agree_contract'] == 'TRUE':
            voting_result = True
        else:
            voting_result = False
        #print(self.name, voting_result)

        return voting_result
    

    def get_action(self, contract, contract_parameter, scope=3):
        '''
            Logic for the agent to decide her action

            Return: action:str
        '''
        world_state, agent_details = self.world.get_world_state()
        # Convert agent details to DataFrame for easier processing
        agent_df = pd.DataFrame(agent_details)
        other_agents = agent_df[agent_df['name'] != self.name]
        # Just collected apples prompting
        agent_sentences = ["{} collected {} apples".format(row['name'], row['just_collected_apple']) for _, row in other_agents.iterrows()]
        collected_apples_sentence = "In the last round, you collected {} apples, {}.".format(self.just_collected_apple, ", ".join(agent_sentences))
        
        # Calculate the nearest agent's coordinates
        agent_df['distance'] = agent_df.apply(lambda row: distance(self.x, self.y, row['x_coord'], row['y_coord']) if row['name'] != self.name else np.inf, axis=1)
        nearest_agent = agent_df.loc[agent_df['distance'].idxmin()]
        nearest_agent_coord = (nearest_agent['x_coord'], nearest_agent['y_coord'])

        # Calculate the nearest apple's coordinates
        nearest_apple_coord = self.nearest_apple(self.x, self.y, world_state)

        # Calculate the number of remaining apples
        remaining_apples = self.count_remaining_apple(world_state)

        # Calculate the number of neighboring apples 
        neighbor_apple = self.world.count_nearby_apples(self.x,self.y,scope) 
        final_contract = contract.replace("X", contract_parameter)
        contract_response = "The contract {contract} is voted yes. This contract will be enforced after every agent takes their actions in this round.".format(contract=final_contract) if self.world.contract_active else ["No contract is enforced this round."]
        
        input_prompt = """
{contract_response} Currently, you are at grid ({x},{y}). The player closet to you is at grid {nearest_agent_coord}. The nearest apple is at grid {nearest_apple_coord}. There are {neighbor_apple} neighboring apples within a radius of {scope} grids around you. In total, there are {remaining_apples} apples. {collected_apples_sentence}

You can choose one of the following actions:
- GO [UP/DOWN/LEFT/RIGHT]: you will move in the following direction for 1 grid.
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
        """.format(
            n_agents=len(agent_details),
            contract_response=contract_response,
            x=self.x,
            y=self.y,
            just_collected_apples=self.just_collected_apple,
            nearest_agent_coord=nearest_agent_coord,
            nearest_apple_coord=nearest_apple_coord,
            scope=scope,
            remaining_apples=remaining_apples,
            neighbor_apple=neighbor_apple,
            collected_apples_sentence=collected_apples_sentence,
            contract=final_contract,
        )

        self.message_history.append(HumanMessage(content=input_prompt))
        print("========>>>>>>")
        print(self.name, self.message_history)
        output = self.call_LLM()
        action = output['action']
        print(self.name, action)
        # TODO generalize reflection into a func
        if(action == "COLLECT" and (not self.world.is_apple_at(self.x,self.y))):
            error_prompt="There's no apple for you to collect in your corrent grid. The nearest apple is at {nearest_apple_coord}. Please reflect and make a correct decision.".format(nearest_apple_coord=nearest_apple_coord)
            self.message_history.append(HumanMessage(content=error_prompt))
            output = self.call_LLM()
            action = output['action']
            print(self.name, "reflected", action)
        print("========>>>>>>")
        self.reset()

        return action


    def call_LLM(self, verbose_input=True, verbose_output=True) -> dict:
        """
        This function takes in the input_prompt and calls LLM.

        Return: Agent decisions in JSON data
        """
        if verbose_input:
            print(f"input_prompt: {self.message_history}")
        _output = self.chat_model(self.message_history)
        json_string = _output.content.split(
            "```json")[-1].strip().replace('```', '')
        try:
            output = json.loads(json_string)
            self.message_history.append(AIMessage(content=json_string))
            # error_execution_message = self.get_error_execution_message(output)
            # if error_execution_message != "":
            #     raise AgentError(error_execution_message)
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
                        content=f"Your output is not in json format. Please make sure your output is in the required json format.")
                self.message_history.append(error_message)
                self.remaining_retry_times -= 1

                return self.call_LLM()
            else:
                raise AgentError(
                    "You have exceeded the maximum number of retries. Please try again later.")

        if verbose_output:
            pprint(output)
        #self.reset()
        return output

    def get_error_execution_message(self, output):
        error_message = []
        agent_list = self.world.agents_map
        for instance in agent_list:
            if(output[instance.name]["order"]):
                order = output[instance.name]["order"]
                # check if the order will cause the agent to move out of the map
                if order.startswith("GO"):
                    target = order.split(" ")[1].lower()
                    if target in ["UP", "DOWN", "LEFT", "RIGHT"]:
                        dir = target
                        x = instance.x + instance.directions[dir.upper()][0]
                        y = instance.y + instance.directions[dir.upper()][1]
                        if x < 0 or x >= self.world.x_size or y < 0 or y >= self.world.y_size:
                            error_message.append(f"Your agent is trying to move out of the map, which is not allowed. ")
            
        error_message.append(HumanMessage(content="Generate new reply based on information above."))
        
        return "\n".join(error_message)


    def _collect_apple(self, x, y):
        if self.world.is_apple_at(x, y):
            self.world.remove_apple(x, y)
            self.rewards += 1
            self.just_collected_apple += 1

    def _move(self, dir):
        new_x = self.x + self.directions[dir][0]
        new_y = self.y + self.directions[dir][1]
        if 0 <= new_x < len(self.world.map[0]) and 0 <= new_y < len(self.world.map):
            self.world.map[self.y][self.x].remove(self)
            self.x, self.y = new_x, new_y
            self.world.map[self.y][self.x].append(self)
        else:
            print(f"Move out of bounds: {dir}")

    def _stay(self):
        pass

    def execute(self, action):
        self.just_collected_apple = 0
        if "GO" in action:
            _, dir = action.split(" ")
            self._move(dir)
        elif "COLLECT" in action:
            self._collect_apple(self.x, self.y)
        elif "STAY" in action:
            self._stay()
