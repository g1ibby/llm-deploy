import typer
from enum import Enum

from llm_deploy.app_logic import AppLogic
from llm_deploy.config import load_config
from llm_deploy.llm_calculator import LLMCalculator
from llm_deploy.utils import print_offer_table, print_instances_table, print_models
from llm_deploy.logging_config import setup_logging

app = typer.Typer()
model_app = typer.Typer()

config = load_config()

# Initialize the business logic with necessary services
appl = AppLogic(config['VAST_API_KEY'], config['LITELLM_API_URL'])

class ChoiceAccess(str, Enum):
    CF = "cf"
    IP = "ip"

def select_offer(gpu_memory: float, disk_space: float, public_ip: bool = True):
    offers = appl.get_offers(gpu_memory, disk_space, public_ip)
    print_offer_table(offers)

    chosen_id = typer.prompt("Which offer do you want to choose?", default=offers[0]["id"], type=int)
    chosen_offer = next((offer for offer in offers if offer['id'] == chosen_id), None)

    if not chosen_offer:
        print("No offer found with the specified ID.")
        return None

    print("You have chosen the following offer:")
    print_offer_table([chosen_offer])
    return chosen_offer

@app.command(help="Applies LLMs configuration based on llms.yaml file")
def apply():
    appl.apply_llms_config()

@app.command(help="Destroys LLMs configuration based on state.json file")
def destroy():
    appl.instance.destroy_all()

@app.command(help="Runs a model with specified parameters like GPU memory, disk space, and access type.")
def run(model: str = typer.Argument(..., help="Model name"),
        gpu_memory: float = typer.Option(0.0, "--gpu-memory", help="GPU memory in GB"),
        disk: float = typer.Option(70.0, "--disk", help="Disk space in GB"),
        access: ChoiceAccess = typer.Option(ChoiceAccess.IP, help="Choose either Cloudflared or IP access")):
    print("Running the model with the following parameters:")
    if gpu_memory == 0.0:
        try:
            calculator = LLMCalculator()
            _, _, model_size = calculator.calculate(model)
            if model_size is None:
                print("Failed to retrieve model size.")
                return
            gpu_memory = model_size
        except Exception as e:
            print(f"Failed to retrieve model size due to an error: {e}")
            return
    typer.echo(f"Running the model: {model} with {gpu_memory} GB of GPU memory. Disk space: {disk} GB. Access: {access}")

    chosen_offer = select_offer(gpu_memory=gpu_memory, disk_space=disk, public_ip=access == ChoiceAccess.IP)
    if not chosen_offer:
        return

    result = appl.run_model(model, chosen_offer['id'], disk_space=disk, public_ip=access == ChoiceAccess.IP)
    if result:
        print("Model run successfully.")
    else:
        print("Failed to run model.")

@app.command(help="Lists all current instances.")
def ls():
    instances = appl.instance.instances()
    print_instances_table(instances)

@app.command(help="Removes an instance by the given ID.")
def rm(id: int = typer.Argument(..., help="Instance ID")):
    chosen_instance = appl.instance.get_instance_by_id(id)
    if chosen_instance:
        print("You have chosen the following instance:")
        print_instances_table([chosen_instance])
    else:
        print("No instance found with the specified ID.")
        return

    appl.instance.destroy_instance(chosen_instance['id'])

@app.command(help="Shows details of an instance by the given ID.")
def show(id: int = typer.Argument(..., help="Instance ID")):
    chosen_instance = appl.instance.get_instance_by_id(id)
    if chosen_instance:
        print_instances_table([chosen_instance])
    else:
        print("No instance found with the specified ID.")
        return
    # Print ollama cloudflared address
    ollama_addr = chosen_instance.get('ollama_addr')
    print(f"Ollama Address: {ollama_addr}")
    # Print models
    models = chosen_instance.get('models')
    if models:
        print("Models on this instance:")
        for model in models:
            name = model.get('name')
            size_in_bytes = model.get('size')
            if size_in_bytes is not None:
                size_in_gb = size_in_bytes / 1e9  # Convert bytes to gigabytes
                print(f"{name}, Size: {size_in_gb:.2f} GB")
            else:
                print(f"{name}, Size: Unknown")

@app.command(help="Retrieves and displays logs for a specified instance ID.")
def logs(id: int = typer.Argument(..., help="Instance ID"), max_logs: int = typer.Option(30, help="Maximum number of logs to retrieve")):
    instance_logs = appl.instance.get_instance_logs(id, max_logs=max_logs)
    if not instance_logs:
        print("Failed to retrieve logs.")
        return
    for log in instance_logs:
        print(log)

@model_app.command(help="Pulls a specified model to a given instance ID.")
def pull(model_name: str = typer.Argument(..., help="Model name"),
         instance_id: int = typer.Argument(..., help="Instance ID")):
    appl.model.model(model_name, instance_id)

@model_app.command(name="rm", help="Removes a specified model from a given instance ID.")
def rm_model(model_name: str = typer.Argument(..., help="Model name"),
             instance_id: int = typer.Argument(..., help="Instance ID")):
    appl.model.model(model_name, instance_id)

@model_app.command(name="ls", help="Lists all models on all instances.")
def ls_models():
    models = appl.model.models()
    print_models(models)


app.add_typer(model_app, name="model", help="Manages model operations like pulling, listing, and removing models.")

def main():
    setup_logging()
    app()

