import os
import requests

BASE = "https://serv.amazingmarvin.com/api/"
TOKEN = os.getenv("AMAZING_MARVIN_TOKEN")


def add_task(title: str):
    """Add a task to the user's Amazing Marvin account.

    Amazing Marvin's markup is availaible to specify the task's properties.
    In particular:
    - Use #parent-name to specify the project/category.
    - Use +3d to schedule it in 3 days (or +2w, +31.12.2023). Use +0d to schedule it today.
    - Use ~5m to set the estimated time.
    - Use HH:MM to set the time of day.
    Note that none of them are required, but add as much information that makes sense.
    """

    url = BASE + "addTask"
    data = {"title": title}
    headers = {
        "X-API-token": TOKEN,
    }
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()

    return "Task added successfully! 📝"


if __name__ == "__main__":
    add_task("Buy milk #groceries +0d ~5m 18:00")
