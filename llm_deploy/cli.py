import typer
from enum import Enum, auto
from pathlib import Path

from llm_deploy.app_logic import AppLogic
from llm_deploy.config import load_config
from llm_deploy.utils import print_offer_table, print_instances_table, print_models
from llm_deploy.logging_config import setup_logging

class OperationMode(Enum):
    CONFIG_MODE = auto()
    MANUAL_MODE = auto()

app = typer.Typer()

# Define subcommand groups
infra_app = typer.Typer(help="Commands for managing infrastructure.")
models_app = typer.Typer(help="Commands for managing models.")

# Add subcommand groups to the main app
app.add_typer(infra_app, name="infra")
app.add_typer(models_app, name="model")

config = load_config()
appl = AppLogic(config['VAST_API_KEY'], config['LITELLM_API_URL'])

CONFIG_MODE_FILE = "llms.yaml"
is_config_mode = Path(CONFIG_MODE_FILE).exists()

def select_offer(gpu_memory: float, disk_space: float, public_ip: bool = True):
    offers = appl.get_offers(gpu_memory, disk_space, public_ip)
    print_offer_table(offers)

    chosen_id = typer.prompt("Which offer do you want to choose?", default=offers[0]["id"], type=int)
    chosen_offer = next((offer for offer in offers if offer['id'] == chosen_id), None)

    if not chosen_offer:
        typer.echo("No offer found with the specified ID.")
        return None

    typer.echo("You have chosen the following offer:")
    print_offer_table([chosen_offer])
    return chosen_offer

def ensure_mode_is(expected_mode: OperationMode):
    current_mode = OperationMode.CONFIG_MODE if is_config_mode else OperationMode.MANUAL_MODE

    if current_mode != expected_mode:
        mode_str = 'config-mode' if expected_mode == OperationMode.CONFIG_MODE else 'manual-mode'
        raise typer.Exit(f"This command is only available in {mode_str}.")

@app.command(help="Applies configuration from llms.yaml. Available in Mode 1.")
def apply():
    ensure_mode_is(OperationMode.CONFIG_MODE)
    typer.echo("Applying llms.yaml configurations...")
    appl.apply_llms_config()

@app.command(help="Destroys infrastructure based on current state. Available in Mode 1.")
def destroy():
    ensure_mode_is(OperationMode.CONFIG_MODE)
    typer.echo("Destroying infrastructure and models...")
    appl.instance.destroy_all()

@infra_app.command(name="ls", help="Lists all machines.")
def infra_ls():
    print_instances_table(appl.instance.instances())

@infra_app.command(name="inspect", help="Shows details for a specified machine.")
def infra_inspect(machine_id: int):
    chosen_instance = appl.instance.get_instance_by_id(machine_id)
    if chosen_instance:
        print_instances_table([chosen_instance])
    else:
        typer.echo("No instance found with the specified ID.")

@infra_app.command(name="create", help="Manually creates a new machine. Available in Mode 2.")
def infra_create(
        gpu_memory: float = typer.Option(0.0, "--gpu-memory", help="GPU memory in GB"),
        disk: float = typer.Option(70.0, "--disk", help="Disk space in GB")):
    ensure_mode_is(OperationMode.MANUAL_MODE)
    chosen_offer = select_offer(gpu_memory, disk, True)
    if not chosen_offer:
        typer.echo("Machine creation cancelled. No offer chosen.")
        return

    typer.echo(f"Creating a machine with: GPU Memory: {gpu_memory} GB, Disk space: {disk} GB")
    try:
        # Assuming `create_instance` has been updated to accept offer_id instead of gpu_memory directly
        appl.instance.create(chosen_offer['id'], disk, True)
        typer.echo("Machine created successfully.")
    except Exception as e:
        typer.echo(f"Failed to create machine due to an error: {e}")

@infra_app.command(name="destroy", help="Destroys a specified machine. Available in Mode 2.")
def infra_destroy(machine_id: int):
    ensure_mode_is(OperationMode.MANUAL_MODE)
    typer.echo(f"Destroying machine {machine_id}...")
    chosen_instance = appl.instance.get_instance_by_id(id)
    if chosen_instance:
        typer.echo("You have chosen the following instance:")
        print_instances_table([chosen_instance])
    else:
        typer.echo("No instance found with the specified ID.")
        return

    appl.instance.destroy_instance(chosen_instance['id'])

@models_app.command(name="deploy", help="Deploys a model to a specified machine. Available in Mode 2.")
def model_deploy(model_name: str, machine_id: int):
    ensure_mode_is(OperationMode.MANUAL_MODE)
    typer.echo(f"Deploying model {model_name} to machine {machine_id}...")
    appl.model.pull(model_name, machine_id)

@models_app.command(name="remove", help="Removes a model from a specified machine. Available in Mode 2.")
def model_remove(model_name: str, machine_id: int):
    ensure_mode_is(OperationMode.MANUAL_MODE)
    typer.echo(f"Removing model {model_name} from machine {machine_id}...")
    appl.model.remove_model(model_name, machine_id)

@models_app.command(name="ls", help="Lists models across machines, or for a specific machine.")
def model_ls():
    print_models(appl.model.models())

@app.command(help="Retrieves and displays logs for a specified machine.")
def logs(machine_id: int, max_logs: int = typer.Option(30)):
    instance_logs = appl.instance.get_instance_logs(machine_id, max_logs=max_logs)
    if instance_logs:
        for log in instance_logs:
            print(log)
    else:
        typer.echo("Failed to retrieve logs.")

def main():
    setup_logging()
    app()

if __name__ == "__main__":
    main()
