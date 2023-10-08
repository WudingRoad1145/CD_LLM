# CD_LLM - Experiments on the Use of Commitment Devices (Formal Contracting) in LLM Agent Games

## API Key Setup Guide

To set up API keys for this project, follow these steps:

1. Create a `llm_api_keys.json` private key JSON file in the `api_key` folder. The content should look something like this:

   ```json
   {
       "openai_api_key_1_wGPT4": "YOUR_KEY",
       "openai_api_key_2": "YOUR_KEY",
       "anthropic_api_key_1": "YOUR_KEY"
   }
2. If it's not set, we will get OPENAI_API_KEY and ANTHROPIC_API_KEY environment variables from your system.


## Experiments progress
- [ ] Static games
  - [ ] Simple Prisoner's Dilemma
    - [ ] Done with zero-shot prompting
  - [ ] Public good
    - [ ] Done with one-shot prompting
- [ ] Dynamic games
  - [ ] Harvest
    - [ ] Game infra set up
    - [ ] Prompting tests finished
    - [ ] CD experiments finished
    - [ ] Stable gaming outcomes w/ CD