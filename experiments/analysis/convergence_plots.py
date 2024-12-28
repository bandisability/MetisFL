import os
import psutil
import torch
import logging


# 1. Dynamic Batch Size Adjustment
def adjust_batch_size(resources, base_batch_size):
    """
    Dynamically adjusts the batch size based on system resource utilization.

    Args:
        resources (dict): System resources including 'cpu' and 'memory'.
        base_batch_size (int): Base batch size for training.

    Returns:
        int: Adjusted batch size.
    """
    cpu_usage = resources.get("cpu", 50)
    memory_usage = resources.get("memory", 50)

    if cpu_usage > 80 or memory_usage > 80:
        return max(1, base_batch_size // 2)  # Reduce batch size if resources are constrained.
    elif cpu_usage < 50 and memory_usage < 50:
        return base_batch_size * 2  # Increase batch size for high-performance devices.
    return base_batch_size


# 2. Adaptive Local Training Steps
def adjust_local_steps(resources, base_local_steps):
    """
    Dynamically adjusts the number of local training steps based on device performance metrics.

    Args:
        resources (dict): System resource utilization metrics.
        base_local_steps (int): Base number of local training steps.

    Returns:
        int: Adjusted number of local training steps.
    """
    cpu_usage = resources.get("cpu", 50)
    memory_usage = resources.get("memory", 50)

    if cpu_usage > 80 or memory_usage > 80:
        return max(1, base_local_steps // 2)  # Reduce local steps for resource-constrained devices.
    elif cpu_usage < 50 and memory_usage < 50:
        return base_local_steps * 2  # Increase local steps for high-performance devices.
    return base_local_steps


# 3. System Resource Monitoring
def monitor_system_resources():
    """
    Monitors system resources such as CPU and memory usage.

    Returns:
        dict: A dictionary containing resource usage metrics with keys 'cpu' and 'memory'.
    """
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    return {"cpu": cpu_usage, "memory": memory_usage}


# 4. Model Snapshot Saving Mechanism
def save_model_snapshot(model, step, snapshot_dir="./snapshots"):
    """
    Saves a snapshot of the model at a specific training step.

    Args:
        model (torch.nn.Module): The model to be saved.
        step (int): The training step.
        snapshot_dir (str): Directory where snapshots are saved.
    """
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir)
    snapshot_path = os.path.join(snapshot_dir, f"model_snapshot_step_{step}.pth")
    torch.save(model.state_dict(), snapshot_path)
    print(f"Model snapshot saved at step {step} to {snapshot_path}")


# 5. Distributed Resource Optimization for Tasks
def distribute_tasks_among_clients(clients, tasks):
    """
    Distributes tasks among clients based on their resource availability.

    Args:
        clients (list): List of client resource dictionaries.
        tasks (list): List of tasks to be distributed.

    Returns:
        dict: A dictionary mapping clients to their allocated tasks.
    """
    task_allocation = {client["id"]: [] for client in clients}
    for task in tasks:
        # Assign tasks to the client with the least CPU utilization.
        client = min(clients, key=lambda c: c["cpu"])
        task_allocation[client["id"]].append(task)
    return task_allocation


# 6. Error Logging
def setup_logging(log_file="training.log"):
    """
    Sets up logging configuration for the training process.

    Args:
        log_file (str): Path to the log file.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def log_error(message):
    """
    Logs an error message.

    Args:
        message (str): Error message to be logged.
    """
    logging.error(message)


# Main Execution for Testing
if __name__ == "__main__":
    # Setup logging configuration
    setup_logging()

    # Simulate system resource monitoring
    resources = monitor_system_resources()
    print(f"System Resources: {resources}")

    # Dynamically adjust batch size
    base_batch_size = 64
    adjusted_batch_size = adjust_batch_size(resources, base_batch_size)
    print(f"Adjusted Batch Size: {adjusted_batch_size}")

    # Dynamically adjust local training steps
    base_local_steps = 10
    adjusted_local_steps = adjust_local_steps(resources, base_local_steps)
    print(f"Adjusted Local Steps: {adjusted_local_steps}")

    # Save model snapshot
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.linear = torch.nn.Linear(10, 1)

    model = DummyModel()
    save_model_snapshot(model, step=5)

    # Simulate distributed task allocation
    clients = [{"id": 1, "cpu": 70}, {"id": 2, "cpu": 30}, {"id": 3, "cpu": 50}]
    tasks = ["task1", "task2", "task3", "task4"]
    task_allocation = distribute_tasks_among_clients(clients, tasks)
    print(f"Task Allocation: {task_allocation}")

    # Log an error for testing purposes
    try:
        1 / 0  # Simulate a runtime error
    except Exception as e:
        log_error(f"An error occurred: {str(e)}")
