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
    model: "g1ibby/miqu:70b"
    priority: low
```

#### Usage

Apply LLMs Configuration: poetry run llm-deploy apply
    Applies configurations from llms.yaml.

Destroy LLMs Configuration: poetry run llm-deploy destroy
    Destroys configurations based on state.json.

Run a Model: poetry run llm-deploy run <model_name> --gpu-memory <memory_in_GB> --disk <disk_space_in_GB> --access <cf/ip>
    Runs a model with specified parameters.

List Current Instances: poetry run llm-deploy ls
    Lists all current instances.

Remove an Instance: poetry run llm-deploy rm <instance_id>
    Removes an instance by ID.

Show Instance Details: poetry run llm-deploy show <instance_id>
    Shows details of an instance by ID.

Retrieve Logs for an Instance: poetry run llm-deploy logs <instance_id> --max-logs <number>
    Retrieves and displays logs for a specified instance.

Model Operations:
    Pull a model: poetry run llm-deploy model pull <model_name> <instance_id>
    Remove a model: poetry run llm-deploy model rm <model_name> <instance_id>
    List models: poetry run llm-deploy model ls

