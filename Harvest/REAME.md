# Dynamic Game - Harvest 
## Start Experimenting
1. In world.py main change agent, round number, world size, scope, etc settings
2. Run the py file and observe Agents' actions in terminal

# Game Logic
## Initialization
1. Init world map with a random density of apples and randomly spawn N agents. Explain the game rules and the initial strategy for each agent.

## Game Rounds
### Global Contracting Mode
2. In each round:
   1. Randomly pick one agent and use the `Propose_Contract` function to ask it whether to propose a CD and decide a parameter:
      - Pass in the global state within the scope.
      - Parse the result and get the parameter X.
   2. If yes, prompt the proposed contract to all players for a round of voting:
      - Pass in the global state within the scope.
      - Propose the CD.
      - Parse the result and record it in a list (need to remember this for each round).
      - If all votes yes, 
         - 1. add the contract in action prompting.
         - 2. activate function punishment in post round
      - If voting failed, simply prompt.
   3. If no, simply prompt for actions:
      - Pass in the global state within the scope.
      - Parse the result into actions.
   4. Execute actions for each agent, updating world states.
   5. If an agent violates the contract, punishment function enforces the committed contract.
   6. Add round memory
   7. Env check(spwan apple, check if game ends(no apple remaining), clear agent current_contract history)
3. Round ends.

### P2P Contracting Mode
2. In each round:
   1. Each agent calls the `Propose_Contract` function to ask it whether to propose a CD, to whom `target_agents`, and decide a parameter.
   2. A voting round within the `target_agents`:
      - All vote YES
         - 1. add the contract in action prompting.
         - 2. activate function punishment in post round
       - If any agent votes NO, ignore
   3. Prompt each agent for actions
   4. Execute actions
   5. Contract enforcement
   6. Add round memory
   7. Env check(spwan apple, check if game ends(no apple remaining), clear agent current_contract history)
3. Round ends

## Prompting Techniques
1. CoT - chain of thoughts - referring to Jason Wei
2. ToM

## Design choices
1. Removed orientation from the initial game as LLM agents are not good at spatial reasoning.
2. Using Unanimity voting for now.
3. Agents see the world state instead of limited by a scope - easier reasoning for now.

## Further augmentations
1. Multiple contracts in force at the same time - concurrency
2. Majority of acceptance as opposed to unanimity
3. Contract overriding only if new contracts are accepted