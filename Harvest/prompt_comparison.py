input_prompt = """
In the last round, you harvested {just_collected_apples} apples. {collected_apples_sentence}
As of now, you are located at grid ({x},{y}), with the nearest competitor at {nearest_agent_coord} and the closest apple at {nearest_apple_coord}. {guide_to_apple} 
Within a {scope}-grid radius, there are {neighbor_apple} apples around you, and a total of {remaining_apples} apples left on the map.

Previous round summary:
{CD_memory}

Current visible world state:
{world_state}

Agents with CD ability available for contracting: {agent_enable_CD}.

You can propose a contract following the template: {contract} Considering the current state and historical trends, would you like to propose a contract to collaboratively manage the apple harvest and prevent overconsumption?

If you propose a contract, it will only be valid for this round and must be agreed upon by all participating agents. Think about the long-term benefits versus the immediate gains, and set the penalty 'X' accordingly.

To propose a contract, provide your reasoning and specify 'X'. Your reasoning should be step-by-step and consider the potential gains or losses, historical contract effectiveness, and the current state of the apple supply. Format your response as follows:
```json
{
    “propose_contract”: “TRUE”,
    “X”: “TODO”,
    "reasoning": "{Your one-line strategic reasoning considering historical data, current game state, and predicted outcomes}"
}

If you believe a contract is not in the best interest this round, please provide your reasoning step by step. Keep your response brief and focused on the game state and your strategy:
```json
{
    “propose_contract”: “FALSE”,
    "reasoning": "{Your one-line reasoning based on game theory, potential risks, and individual strategy}"
}

""".format(
        strategy=self.strategy,
        n_agents=len(agent_details),
        contract=contract,
        x=self.x,
        y=self.y,
        just_collected_apples=self.just_collected_apple,
        world_state=world_state,
        nearest_agent_coord=nearest_agent_coord,
        nearest_apple_coord=nearest_apple_coord,
        scope=scope,
        remaining_apples=remaining_apples,
        neighbor_apple=neighbor_apple,
        collected_apples_sentence=collected_apples_sentence,
        CD_memory=memory_sentence,
        guide_to_apple=guide_to_apple,
        agent_enable_CD=agent_enable_CD,
    )