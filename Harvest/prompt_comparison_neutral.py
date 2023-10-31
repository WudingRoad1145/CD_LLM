contract_prompt = """
Last round's harvest: {collected_apples_sentence}
Current location: grid ({x},{y}). Nearest competitor location: {nearest_agent_coord}. Nearest apple location: {nearest_apple_coord}. {guide_to_apple} 
Apples within a {scope}-grid radius: {neighbor_apple}. Total apples remaining: {remaining_apples}.

Contract history:
{CD_memory}

Visible world state:
{world_state}

Agents with CD ability: {agent_enable_CD}.

Proposal for a contract is an available action. It must follows the template: {contract} It will be enforced for only one round if all agents with CD ability agree and is only effective on agents with CD ability.

If proposing a contract, define the variable 'X' and provide reasoning step by step. Ensure the response is formatted and reasoned within one line:
```json
{
    “propose_contract”: “TRUE”,
    “X”: “{value}”,
    "reasoning": "{concise reasoning}"
}

If not proposing a contract, provide reasoning step by step. Response should be one line:
```json
{
    “propose_contract”: “FALSE”,
    "reasoning": "{concise reasoning}"
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

voting_prompt="""
Observation: Location: grid ({x},{y}). Closest player: grid {nearest_agent_coord}. Closest apple: grid {nearest_apple_coord}. {guide_to_apple} 
Local apple count within {scope} grids: {neighbor_apple}. Total apples: {remaining_apples}.
Last round's collection: {just_collected_apples} apples. {collected_apples_sentence}

Visible world state:
{world_state}

Contract history:
{CD_memory}

Agents eligible for CD participation: {agent_enable_CD}.

Proposal received from {proposer}: {contract}

Your response to the contract proposal is required. Formulate your decision and reasoning concisely. Respond in one line using the following format:

```json
{
    “agree_contract”: “{decision}”,
    "reasoning": "{your reasoning}"
}


""".format(
        n_agents=len(agent_details),
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
        contract=final_contract,
        proposer=proposer_name,
        CD_memory=memory_sentence,
        guide_to_apple=guide_to_apple,
        agent_enable_CD=agent_enable_CD,
    )