from prettytable import PrettyTable
import datetime

def format_ram(ram_in_mb):
    """Helper function to convert RAM from MB to GB and format it as a string."""
    if ram_in_mb is not None and ram_in_mb != 'N/A':
        return f"{ram_in_mb / 1024:.1f} GB"
    return 'N/A'

def format_price(price):
    """Helper function to format the price as a string with '$/h' suffix."""
    if price is not None and price != 'N/A':
        return f"{float(price):.3f}$/h"
    return 'N/A'

def format_flops(flops):
    """Helper function to format the FLOPS value to one decimal place."""
    if flops is not None and flops != 'N/A':
        return f"{float(flops):.1f}"
    return 'N/A'

def format_time(start_timestamp):
    """Helper function to format elapsed time from a start timestamp to now."""
    if start_timestamp is not None and start_timestamp != 'N/A':
        start_time = datetime.datetime.fromtimestamp(start_timestamp)
        now = datetime.datetime.now()
        elapsed = now - start_time

        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, _ = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"
    return 'N/A'

def print_offer_table(offers):
    table = PrettyTable()
    table.field_names = ["ID", "GPU Info", "Total GPU RAM", "CPU/RAM", "Total FLOPS", "Price"]
    table.align["GPU Info"] = "l"  # Left aligns the 'GPU Info' column

    for offer in offers:
        num_gpus = offer.get('num_gpus', 1)
        gpu_ram = format_ram(offer.get('gpu_ram'))
        gpu_info = f"{offer.get('gpu_name', 'N/A')} / {gpu_ram}"
        if num_gpus != 'N/A' and num_gpus > 1:
            gpu_info = f"{num_gpus}x{gpu_info}"

        gpu_totalram = format_ram(offer.get('gpu_totalram'))
        cpu_info = f"{offer.get('cpu_name', 'N/A')} / {format_ram(offer.get('cpu_ram'))}"
        total_flops = format_flops(offer.get('total_flops'))
        price = format_price(offer.get('dph_total'))
        numeric_id = offer.get('id', 'N/A')

        table.add_row([numeric_id, gpu_info, gpu_totalram, cpu_info, total_flops, price])

    print(table)

def print_instances_table(instances):
    table = PrettyTable()
    table.field_names = ["ID", "Host Run Time", "GPU Info", "GPU RAM Usage", "CPU/RAM", "Total FLOPS", "Price", "Disk", "Internet"]
    table.align["GPU Info"] = "l"  # Left aligns the 'GPU Info' column

    for instance in instances:
        numeric_id = instance.get('id', 'N/A')

        host_run_time = format_time(instance.get('start_date'))

        num_gpus = instance.get('num_gpus', 1)
        gpu_ram = format_ram(instance.get('gpu_ram'))
        gpu_info = f"{instance.get('gpu_name', 'N/A')} / {gpu_ram}"
        if num_gpus != 'N/A' and num_gpus > 1:
            gpu_info = f"{num_gpus}x{gpu_info}"

        vmem_usage = instance.get('vmem_usage', 'N/A')
        if isinstance(vmem_usage, float):
            vmem_usage = f"{vmem_usage:.1f} GB"

        cpu_info = f"{instance.get('cpu_name', 'N/A')} / {format_ram(instance.get('cpu_ram'))}"
        total_flops = format_flops(instance.get('total_flops'))
        price = format_price(instance.get('dph_total'))

        # New fields for Disk and Internet
        disk_util = instance.get('disk_util', 'N/A')
        if isinstance(disk_util, float):
            disk_util = f"{disk_util:.1f}"
        disk_space = instance.get('disk_space', 'N/A')
        disk_info = f"{disk_util} / {disk_space}"

        inet_up = instance.get('inet_up', 'N/A')
        inet_down = instance.get('inet_down', 'N/A')
        internet_info = f"{inet_up} / {inet_down}"

        table.add_row([numeric_id, host_run_time, gpu_info, vmem_usage, cpu_info, total_flops, price, disk_info, internet_info])

    print(table)

def print_pull_status(pull_model_generator):
    for status in pull_model_generator:
        if 'status' in status:
            if status['status'] == 'pulling manifest':
                print('pulling manifest')
            elif 'digest' in status:
                total_size = status['total']
                completed = status.get('completed', 0)
                percentage = (completed / total_size) * 100 if total_size else 0
                bar_length = 50
                filled_length = int(bar_length * percentage // 100)
                bar = '▕' + '█' * filled_length + '-' * (bar_length - filled_length) + '▏'
                print(f"\r {status['status']}... {percentage:.2f}% {bar} ({completed/1e9:.1f} GB/{total_size/1e9:.1f} GB)", end="")
            elif status['status'] == 'success':
                print("\nDownload completed successfully.")

def print_models(models):
    table = PrettyTable()
    table.field_names = ["Model Name", "Instance"]
    table.align["Model Name"] = "l"  # Left aligns the 'Model Name' column

    for model in models:
        model_name = model.get('name', 'N/A')
        instance_id = model.get('instance_id', 'N/A')

        table.add_row([model_name, instance_id])

    print(table)

