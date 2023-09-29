# Dynamic Game - Harvest Game Logic

## Initialization
1. Init world map with a random density of apples and randomly spawn agents.

## Game Rounds
2. In each round:
   1. Randomly pick one agent and use the `Propose_Contract` function to ask it to decide a parameter:
      - Explain the game rules.
      - Q: Whether to pass that in every round?
      - Pass in the global state within the scope.
      - Parse the result and get the parameter X.
   2. If yes, prompt the proposed contract to all players for a round of voting:
      - Explain the game rules.
      - Pass in the global state within the scope.
      - Propose the CD.
      - Parse the result and record it in a list (need to remember this for each round).
      - If the majority votes yes, add the contract in action prompting.
      - If voting failed, simply prompt.
   3. If no, simply prompt:
      - Pass in the global state within the scope.
      - Parse the result into actions.
   4. Execute actions for each agent, updating world states.
   5. Spawn new apples according to the new world states; the round ends.
