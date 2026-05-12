import os
import sys
import logging
import requests
import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import re

# Setup Logging
logger = logging.getLogger("CodingAgent")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

fh = logging.FileHandler('agent_run.log', encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

class AgentTools:
    @staticmethod
    def read_file(path: str) -> str:
        try:
            return Path(path).read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file {path}: {e}"

    @staticmethod
    def write_file(args: str) -> str:
        # Format expects: path\n\ncontent
        if "\n\n" not in args:
            return "Error: Invalid input format. Expected: <path>\\n\\n<content>"
        try:
            path, content = args.split("\n\n", 1)
            p = Path(path.strip())
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Successfully wrote to {p}"
        except Exception as e:
            return f"Error writing file: {e}"

    @staticmethod
    def list_directory(path: str = ".") -> str:
        try:
            p = Path(path.strip())
            if not p.exists():
                return f"Error: Directory not found: {p}"
            if not p.is_dir():
                return f"Error: Not a directory: {p}"
            entries = sorted(child.name for child in p.iterdir())
            return "\n".join(entries) if entries else "<empty>"
        except Exception as e:
            return f"Error listing directory: {e}"

    @staticmethod
    def execute_command(command: str) -> str:
        try:
            result = subprocess.run(command, shell=True, text=True, capture_output=True, timeout=60)
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            return output if output else "<command executed silently>"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 60 seconds."
        except Exception as e:
            return f"Error executing command: {e}"

class CodingAgent:
    def __init__(self, endpoint: str, max_steps: int = 15):
        self.endpoint = endpoint
        self.max_steps = max_steps
        self.tools = {
            "read_file": AgentTools.read_file,
            "write_file": AgentTools.write_file,
            "list_directory": AgentTools.list_directory,
            "execute_command": AgentTools.execute_command
        }
        
        self.system_prompt = """You are an autonomous coding agent capable of planning and executing tasks using tools.
You work in a loop of Thought, Action, Action Input, and Observation.
At the beginning, you should break down the user's task into a plan.

Available tools:
1. read_file: Reads the contents of a file. Action Input format: <absolute-or-relative-path>
2. write_file: Writes text to a file. Action Input format: <path>\\n\\n<content>
3. list_directory: Lists the contents of a directory. Action Input format: <path>
4. execute_command: Executes a bash command and returns the output. Action Input format: <command>

To use a tool, you MUST use the exact following format:
Thought: <reasoning about what to do next>
Action: <the name of the tool to use>
Action Input: <the exact input for the tool>

After you output "Action Input:", STOP generation. The system will provide the "Observation:" of the tool.
Do NOT generate the Observation yourself.

When you are completely finished with the task and have verified it, output the final result using the format:
Thought: I have completed the task.
Final Answer: <your final answer or summary>
"""
        self.history = [{"role": "system", "content": self.system_prompt}]

    def _call_llm(self) -> str:
        lines = []
        for msg in self.history:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                lines.append(f"system: {content}")
            elif role == "user":
                lines.append(f"user: {content}")
            elif role == "assistant":
                lines.append(f"assistant: {content}")
            elif role == "tool":
                lines.append(f"assistant: Observation: {content}")
        
        if self.history[-1]["role"] != "assistant":
            lines.append("assistant:")
            
        prompt = "\n".join(lines)

        logger.debug("--- Prompt sent to LLM ---")
        logger.debug(prompt)
        logger.debug("--------------------------")
        
        payload = {
            "prompt": prompt,
            "max_length": 1024,
            "temperature": 0.2
        }
        try:
            resp = requests.post(self.endpoint, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            # The model might return the entire prompt + generation. Let's assume it returns just the generation or we need to extract it.
            text = data.get("text", "")
            
            # If the model echoes the prompt, strip it
            if text.startswith(prompt):
                text = text[len(prompt):]
                
            return text
        except Exception as e:
            logger.error(f"Error calling LLM endpoint: {e}")
            return f"Error: {e}"

    def run(self, task: str):
        logger.info(f"Starting task: {task}")
        self.history.append({"role": "user", "content": task})

        for step in range(self.max_steps):
            logger.info(f"--- Step {step + 1} ---")
            
            response = self._call_llm()
            
            # We must aggressively truncate the response to prevent hallucinated observations
            # Find the first Action Input: block and cut off anything after it that looks like an observation.
            observation_idx = response.find("Observation:")
            if observation_idx != -1:
                response = response[:observation_idx]

            logger.info(f"LLM Response:\n{response}")
            self.history.append({"role": "assistant", "content": response.strip()})

            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:", 1)[1].strip()
                logger.info(f"Task Completed. Final Answer: {final_answer}")
                return final_answer

            # Parse Action and Action Input
            action_match = re.search(r"Action:\s*(.*?)\s*\n", response)
            action_input_match = re.search(r"Action Input:\s*(.*)", response, re.DOTALL)

            if action_match and action_input_match:
                action = action_match.group(1).strip()
                action_input = action_input_match.group(1).strip()
                
                # Further cleanup of action_input in case it hallucinated extra stuff
                if "\nThought:" in action_input:
                     action_input = action_input.split("\nThought:")[0].strip()

                logger.info(f"Executing Tool -> {action}")
                logger.debug(f"Tool Input -> {action_input}")

                if action in self.tools:
                    observation = self.tools[action](action_input)
                else:
                    observation = f"Error: Tool '{action}' not found. Available tools: {list(self.tools.keys())}"
                
                logger.info(f"Observation:\n{observation}")
                self.history.append({"role": "tool", "content": str(observation)})
            else:
                logger.warning("Could not parse Action and Action Input from response. Prompting model to use correct format.")
                self.history.append({"role": "tool", "content": "Error: You must provide an Action and Action Input in the correct format, or output Final Answer: if finished."})
        
        logger.error("Max steps reached without finding Final Answer.")
        return "Task failed: Max steps reached."

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Coding Agent")
    parser.add_argument("--task", type=str, required=True, help="The task for the agent to perform.")
    args = parser.parse_args()

    # Try to load .env for endpoint
    try:
        load_dotenv(encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv(encoding="utf-16")
        
    endpoint = os.environ.get("GEMMA_ENDPOINT")
    if not endpoint:
        logger.error("GEMMA_ENDPOINT environment variable not set. Please set it in .env or environment.")
        sys.exit(1)

    agent = CodingAgent(endpoint=endpoint)
    agent.run(args.task)
