import os
import time
import multiprocessing as mp
from openai import OpenAI
from dotenv import load_dotenv

from prover.utils import AttrDict, MODEL_FORMAT

class GeneratorProcess(mp.Process):
    def __init__(self, local_rank, node_rank, model_name, task_queue, request_statuses, lock, args):
        super().__init__()
        self.local_rank = local_rank
        self.node_rank = node_rank
        self.model_name = model_name
        self.task_queue = task_queue
        self.request_statuses = request_statuses
        self.lock = lock
        self.args = args
        self.prompt_func = MODEL_FORMAT[args.mode]['prompt']
        self.output_func = MODEL_FORMAT[args.mode]['output']

        # Load environment variables
        load_dotenv()

        # Set up the OpenAI client
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY")
        )

    def run(self):
        while True:
            inputs = self.task_queue.get()
            if inputs is None:  # Terminate when receiving None
                break
            
            model_inputs = [
                ''.join([
                    item.get('_extra_header', str()),
                    self.prompt_func(item),
                    item.get('_extra_prompt', str()),
                ]) for _, _, item in inputs
            ]
            
            outputs = []
            for input_text in model_inputs:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": input_text}],
                        temperature=self.args.temperature,
                        max_tokens=self.args.max_tokens,
                        top_p=self.args.top_p,
                    )
                    output = self.output_func(response.choices[0].message.content)
                    outputs.append(output)
                except Exception as e:
                    print(f"Error in API call: {e}")
                    outputs.append(None)

            with self.lock:
                for (_, request_id, _), output in zip(inputs, outputs):
                    self.request_statuses[request_id] = output