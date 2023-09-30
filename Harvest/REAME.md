# Dynamic Game - Harvest Game Logic

## Initialization
1. Init world map with a random density of apples and randomly spawn N agents.

## Game Rounds
2. In each round:
   1. Randomly pick one agent and use the `Propose_Contract` function to ask it whether to propose a CD and decide a parameter:
      - Explain the game rules.
        - Q: Whether to pass that in every round?
      - Pass in the global state within the scope.
      - Parse the result and get the parameter X.
   2. If yes, prompt the proposed contract to all players for a round of voting:
      - Explain the game rules.
      - Pass in the global state within the scope.
      - Propose the CD.
      - Parse the result and record it in a list (need to remember this for each round).
      - If all votes yes, 
         - 1. add the contract in action prompting.
         - 2. activate function punishment in post round
      - If voting failed, simply prompt.
   3. If no, simply prompt:
      - Pass in the global state within the scope.
      - Parse the result into actions.
   4. Execute actions for each agent, updating world states.
   5. If an agent violates the contract, punishment function enforces the committed contract.
   6. Spawn new apples according to the new world states.
3. Round ends.


## Design choices
1. Removed orientation from the initial game as LLM agents are not good at spatial reasoning.
2. Using Unanimity voting for now.
3. Agents see the world state instead of limited by a scope - easier reasoning for now.

## Further augmentations
1. Multiple contracts in force at the same time - concurrency
2. Majority of acceptance as opposed to unanimity
3. Contract overriding only if new contracts are accepted