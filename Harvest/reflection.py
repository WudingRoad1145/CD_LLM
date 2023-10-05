class Reflection:
    def __init__(self, memory):
        self.memory = memory

    def reflect_on_contract(self):
        '''
            Reflect on the agent's contracts.
        '''
        proposer = self.memory['proposer']
        recent_contract = self.memory['contract_proposed'] if self.memory['contract_proposed'] is not None else False
        voting_results = self.memory['voting_results']
        rewards = self.memory['agent_rewards'][self.name]
        recent_action = self.memory["exec_results"]
        contract_enforcement_results = self.memory['contract_enforcement_results']
        distributed_rewards = self.memory['distributed_rewards']

        # Analyze the contract's effect on rewards
        beneficial_to_agent = True if distributed_rewards[self.name] >= 0 else False
        #improved_social_welfare = TODO remaining apples after contract > remaining apples before contract

        if self.name == proposer:
            # Analyze the contract from the proposer's perspective
            reflection_template = """
                                You proposed a contract: {contract} which was {status}. {voting_results} 
                                {contract_enforcement_results}
                                {beneficial_outcome} 
                                Do you think proposing the contract was beneficial? How would you propose a better contract? Please reflect on your contracting step by step.
                                """
            reflection = reflection_template.format(
                contract=recent_contract,
                status='passed. Your action last round was {} and your reward is {}. How did the contract affect your action?'.format(recent_action[self.name], rewards) if self.world.contract_active else 'rejected. Please reflect on why some agents rejected your contract. Is it because the parameter was set unreasonable?',
                voting_results=voting_results,
                contract_enforcement_results=contract_enforcement_results,
                beneficial_outcome='The contract was enforced and was beneficial to yourself.' if beneficial_to_agent and self.world.contract_active else 'The contract was enforced and was not beneficial to yourself.'
            )
            self.message_history.append(reflection)
        else:
            # Analyze the contract from the contract voter's perspective
            reflection_template = """
                                You voted on a contract proposed by {proposer}: {contract}. 
                                The contract was {status}. 
                                {voting_results}
                                {contract_enforcement_results}
                                {beneficial_outcome}
                                Reflect on why you voted the way you did. Was it in your best interest? How did the contract affect your actions and rewards? What would you propose if you are the proposer? Please reflect step by step.
                                """
            reflection = reflection_template.format(
                proposer=proposer,
                contract=recent_contract,
                status='accepted and enforced. Your action last round was {} and your reward is {}. How did the contract affect your action?'.format(recent_action[self.name], rewards) if self.world.contract_active else 'rejected. Reflect on why the majority voted this way. Was the contract not beneficial for most?',
                voting_results=voting_results,
                contract_enforcement_results=contract_enforcement_results,
                beneficial_outcome='The contract was enforced and was beneficial to you.' if beneficial_to_agent and self.world.contract_active else 'The contract was enforced and was not beneficial to you.'
            )
            self.message_history.append(reflection)

        self.call_LLM()

    def reflect_on_actions(self):
        '''
            Reflect on the agent's actions.
        '''
        rewards = self.memory['agent_rewards'][self.name]
        recent_action = self.memory["exec_results"][self.name]

        reflection_template = """
                          Your recent action was {recent_action} and you collected {reward} apple.
                          The world state is {world_state}.
                          Do you think you could have made a better action? How would you have done it? How can you improve in this round? Please reflect on your actions step by step.
                          """
    
        # TODO Analyze how you could have improved your rewards and social welfare. This might be RL
        # potential_reward_improvement = max(self.memory['potential_rewards']) - self.memory['agent_rewards']['self']
        # reward_improvement = f"You could have improved your rewards by {potential_reward_improvement}." if potential_reward_improvement > 0 else ""

        reflection = reflection_template.format(
            recent_action=recent_action,
            #reward_improvement=reward_improvement,
            reward = rewards,
        )
        self.message_history.append(reflection)
        
        self.call_LLM()
