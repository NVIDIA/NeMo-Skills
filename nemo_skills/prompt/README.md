This folder contains a collection of prompts used in our codebase. The majority of the prompts can be found
inside `config/generic` folder and are meant to be used by default.

In some cases we also have model-specific prompts if they were identified to work much better for a specific
model family or are required to reproduced published evals (e.g. see `config/llama3` folder).

Finally, there is a `config/judge` `folder which contains prompts for various LLM-as-a-judge use-cases.