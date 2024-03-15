### LLM deploy - tool to manage your LLMs on vast.ai servers

#### Introduction
"llm-deploy" is a Python tool for deploying and managing large language models (LLMs) on vast.ai using ollama. It uses Typer for command-line interactions.

#### Requirements
- Python 3.11 or later
- Poetry for dependency management

#### Installation
1. Clone the repository or download the source code.
2. Navigate to the project directory.
3. Run `poetry install` to install dependencies.

#### Configuration
Create a `llms.yaml` file with your model configurations, like this:
```yaml
models:
  llama:
    model: "phi:2.7b-chat-v2-q5_K_M"
    priority: low
```

Copy file `env.sh.dist` to `env.sh` and set your keys there. 

Run `source env.sh` 

### Usage

#### Config-Mode Commands:

- Apply LLMs Configuration:
`poetry run llm-deploy apply`
    Applies configurations from llms.yaml.

- Destroy LLMs Configuration:
`poetry run llm-deploy destroy`
    Reverts configurations and destroys created instances based on the current state.

#### Manual-Mode Commands:

- List Current Instances:
`poetry run llm-deploy infra ls`
    Lists all current instances.

- Create New Instance (Manual):
`poetry run llm-deploy infra create --gpu-memory <memory_in_GB> --disk <disk_space_in_GB>`
    Manually creates a new instance with specified GPU memory, disk space, and public IP option.

- Remove an Instance:
`poetry run llm-deploy infra destroy <instance_id>`
    Removes an instance by ID.

- Show Instance Details:
`poetry run llm-deploy infra inspect <instance_id>`
    Shows details of an instance.

- Retrieve Logs for an Instance:
`poetry run llm-deploy logs <instance_id> --max-logs <number>`
    Retrieves and displays logs for a specified instance.

- Deploy a Model to an Instance:
`poetry run llm-deploy model deploy <model_name> <instance_id>`
    Deploys a specified model to an instance.

- Remove a Model from an Instance:
`poetry run llm-deploy model remove <model_name> <instance_id>`
    Removes a deployed model from an instance.

- List Models on Instances:
`poetry run llm-deploy model ls`
    Lists models deployed across instances.
