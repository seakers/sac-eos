import json
import sys
import argparse
import traceback

from scripts.sac import SoftActorCritic
from scripts.utils import DataFromJSON
from scripts.client import Client

if __name__ == "__main__":
    try:
        # Gather the arguments
        argparse = argparse.ArgumentParser()

        argparse.add_argument("--host", default="localhost", type=str, help="Host address.")
        argparse.add_argument("--port", default=5555, type=int, help="Port number.")
        argparse.add_argument("--save", type=str, help="Configuration file.")

        args = argparse.parse_args()

        # Create agent
        client = Client(gym_host=args.host, gym_port=args.port)

        # Load the configuration file
        with open(f"{sys.path[0]}/sac-configuration.json", "r") as file:
            config = json.load(file)

        # Create configuration object
        conf = DataFromJSON(config, "configuration")

        # Create the SAC algorithm
        sac = SoftActorCritic(conf, client, args.save)

        # Start the SAC algorithm
        sac.start()
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

    finally:
        sac.client.shutdown_gym()
        