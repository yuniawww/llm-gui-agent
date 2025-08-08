"""Script to run evaluation using my custom grounding agent.
Based on the original OS-World run.py architecture.
"""


import argparse
import datetime
import json
import logging
import os
import sys
import requests
from tqdm import tqdm

import lib_run_single
from desktop_env.desktop_env import DesktopEnv
from mm_agents.my_grounding_agent import OSWorldGroundingAgent

# Logger Setup
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

# Setup log handlers
file_handler = logging.FileHandler(
    os.path.join("logs", f"normal-{datetime_str}.log"), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", f"debug-{datetime_str}.log"), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler(
    os.path.join("logs", f"sdebug-{datetime_str}.log"), encoding="utf-8"
)

# Set log levels
file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG)

# Format logs
formatter = logging.Formatter(
    "\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
sdebug_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))
sdebug_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger.addHandler(sdebug_handler)

logger = logging.getLogger("desktopenv.experiment")

SERVER_URL = "http://127.0.0.1:5000" 

def config():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run evaluation using my custom grounding agent"
    )

    # Environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless mode"
    )
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=15)
    
    
    parser.add_argument("--vm_ip", type=str, help="IP address of the VM")
    parser.add_argument("--server_port", type=int, default=5000, help="Port for the server")

    # Example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples"
    )
    parser.add_argument(
        "--test_all_meta_path", type=str, default="evaluation_examples/test_all.json"
    )

    # Logging config
    parser.add_argument("--result_dir", type=str, default="./results")

    return parser.parse_args()



def call_setup_api(server_url: str, endpoint: str, payload: dict) -> bool:
    """
    A general function to call the server's setup API (sending JSON data).
    """
    api_url = f"{server_url}{endpoint}"
    try:
        response = requests.post(api_url, json=payload, timeout=150)
        response.raise_for_status()
        result = response.json()
        logger.info(f"  > API {endpoint} called successfully.")
        return result.get('success', False)
    except requests.exceptions.RequestException as e:
        logger.error(f"  > Error: Failed to call API {endpoint}: {e}")
        if e.response is not None:
            logger.error(f"  > Server response details (Status {e.response.status_code}):\n{e.response.text}")
        return False

def run_setup_for_example(example_config: dict, config_file_path: str, server_url: str) -> bool:
    """
    Executes all setup tasks based on the 'config' section of a single test case.
    """
    setup_tasks = example_config.get('config', [])
    if not setup_tasks:
        logger.info("No 'config' tasks for this test case, skipping environment setup.")
        return True

    logger.info("--- Executing Environment Setup ---")
    
    for i, task in enumerate(setup_tasks, 1):
        task_type = task.get('type')
        parameters = task.get('parameters')
        logger.info(f"  - Processing task {i}/{len(setup_tasks)}: Type '{task_type}'")

        success = False
        if task_type == 'download':
            payload = parameters.get("files")
            if payload is None:
                logger.error("  > Error: 'files' key is missing in the parameters for the 'download' task.")
                return False
            success = call_setup_api(server_url, "/setup/download_from_url", payload)
        
        elif task_type == 'launch':
            success = call_setup_api(server_url, "/setup/launch", parameters)
        
        elif task_type == 'execute':
            if "command" not in parameters:
                logger.error("  > Error: 'command' key is missing in the parameters for the 'execute' task.")
                return False
            success = call_setup_api(server_url, "/setup/execute", parameters)
        
        else:
            logger.warning(f"  - Warning: Skipping unknown setup task type '{task_type}'.")
            continue 
        
        if not success:
            logger.error(f"Setup task '{task_type}' failed. Aborting evaluation for this test case.")
            return False

    logger.info("--- Environment Setup Successful ---")
    return True

def test(args, test_all_meta):
    """Run evaluation on test examples."""
    scores = []
    max_steps = args.max_steps

    # Log arguments
    logger.info("Args: %s", args)

    vm_ip = args.vm_ip
    if not vm_ip:
        logger.warning("No VM IP provided, attempting to use localhost")
        vm_ip = "localhost"
        
    logger.info(f"Connecting to VM at {vm_ip}:{args.server_port}")
    server_url = f"http://{vm_ip}:{args.server_port}"

    # Initialize agent and environment
    agent = OSWorldGroundingAgent(vm_ip=args.vm_ip, server_port=args.server_port)

    env = DesktopEnv(
        action_space="pyautogui",
        screen_size=(args.screen_width, args.screen_height),
        headless=False,
        os_type="Ubuntu",
        require_a11y_tree=True,
        vm_ip=vm_ip,
        server_port=args.server_port
    )

    # Run evaluation loop
    for domain in tqdm(test_all_meta, desc="Domain"):
        for example_id in tqdm(test_all_meta[domain], desc="Example", leave=False):
            config_file_path = os.path.join(
                args.test_config_base_dir, f"examples/{domain}/{example_id}.json"
            )
            try:
                with open(config_file_path, "r", encoding="utf-8") as f:
                    example = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read config file {config_file_path}: {e}")
                continue

            logger.info(f"===== Starting Test: {domain}/{example_id} =====")
            logger.info(f"[Instruction]: {example['instruction']}")

            setup_success = run_setup_for_example(example, config_file_path, server_url)
            
            if not setup_success:
                logger.error(f"Environment setup failed, skipping evaluation: {example_id}")
                continue

            # Create result directory
            example_result_dir = os.path.join(
                args.result_dir,
                "pyautogui",  # Our agent's action space
                "screenshot_a11y_tree",  # Our agent's observation type
                "my_grounding_agent",
                domain,
                example_id,
            )
            os.makedirs(example_result_dir, exist_ok=True)

            try:
                # Run single example
                lib_run_single.run_single_example(
                    agent,
                    env,
                    example,
                    max_steps,
                    example["instruction"],
                    args,
                    example_result_dir,
                    scores,
                    domain=domain,
                )
            except Exception as e:
                logger.error(f"Exception in {domain}/{example_id}: {e}")
                try:
                    env.controller.end_recording(
                        os.path.join(example_result_dir, "recording.mp4")
                    )
                except Exception as rec_error:
                    logger.error(f"Error during recording cleanup: {rec_error}")
                    
                if len(scores) == 0:
                    scores.append(0.0)  
                    
                with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                    f.write(
                        json.dumps(
                            {"Error": f"Error in {domain}/{example_id}: {str(e)}"}
                        )
                    )
                    f.write("\n")

    env.close()
    logger.info(f"Average score: {sum(scores) / len(scores)}")

def get_unfinished(result_dir, total_file_json):
    """Get list of unfinished test examples."""
    target_dir = os.path.join(
        result_dir, 
        "pyautogui",
        "screenshot_a11y_tree", 
        "my_grounding_agent"
    )

    if not os.path.exists(target_dir):
        return total_file_json

    finished = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" not in os.listdir(example_path):
                        # Clear incomplete results
                        for file in os.listdir(example_path):
                            os.remove(os.path.join(example_path, file))
                    else:
                        finished[domain].append(example_id)

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [
                x for x in total_file_json[domain] if x not in examples
            ]

    return total_file_json

def get_result(result_dir, total_file_json):
    """Get current evaluation results."""
    target_dir = os.path.join(
        result_dir,
        "pyautogui",
        "screenshot_a11y_tree",
        "my_grounding_agent"
    )
    
    if not os.path.exists(target_dir):
        print("New experiment, no results yet.")
        return None

    all_results = []

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        try:
                            score = float(
                                open(os.path.join(example_path, "result.txt"), "r").read()
                            )
                            all_results.append(score)
                        except:
                            all_results.append(0.0)

    if not all_results:
        print("New experiment, no results yet.")
        return None

    success_rate = sum(all_results) / len(all_results) * 100
    print(f"Current Success Rate: {success_rate:.2f}%")
    return all_results

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = config()

    
    args.headless = False  
    args.path_to_vm = None  

    test_all_meta = {
        "os": ["28cc3b7e-b194-4bc9-8353-d04c0f4d56d2"]
    }

    # Load test examples
    # with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
    #     test_all_meta = json.load(f)

    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    # Get unfinished examples
    test_file_list = get_unfinished(args.result_dir, test_all_meta)
    
    # Log remaining tasks
    left_info = ""
    for domain in test_file_list:
        left_info += f"{domain}: {len(test_file_list[domain])}\n"
    logger.info(f"Left tasks:\n{left_info}")

    # Get current results
    get_result(args.result_dir, test_all_meta)
    
    # Run evaluation
    print("--- RUNNING STEP 3: OPEN APP TEST ---")
    test(args, test_file_list)