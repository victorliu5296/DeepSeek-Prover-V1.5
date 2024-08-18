import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

from prover.utils import get_datetime, load_jsonl_objects, MODEL_FORMAT

class SamplingAlgorithmBase(object):
    def __init__(self, scheduler, model_name, process_print, cfg, **kwargs):
        # Load environment variables
        load_dotenv()

        self.scheduler = scheduler
        self.model_name = model_name
        self.process_print = process_print
        self.cfg = cfg

        # Set up the OpenAI client
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY")
        )

        self.max_tokens = cfg.max_tokens
        self.few_shot_dataset = cfg.get('few_shot_dataset', None)
        if self.few_shot_dataset is not None:
            self.few_shot_dataset = load_jsonl_objects(self.few_shot_dataset)
        self.few_shot_num = cfg.get('few_shot_num', 3)
        self.few_shot_func = MODEL_FORMAT[cfg.mode]['few_shot']
        self.log_interval = cfg.get('log_interval', 32)
    
    @property
    def algorithm_name(self):
        return self.__class__.__name__
    
    def _post_sample_info(self, **kwargs):
        return dict(
            algorithm=self.algorithm_name,
            datetime=get_datetime(),
            **kwargs,
        )
    
    def _encode_length(self, code):
        # Use the API to get token count
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": code}],
            max_tokens=1,  # We only need the token count, not actual completion
        )
        return response.usage.prompt_tokens
    
    def _preprocess_data(self, input_data):
        if self.few_shot_dataset is None or self.few_shot_num == 0:
            return input_data
        return {
            **input_data,
            '_extra_header': ''.join([
                self.few_shot_func(self.few_shot_dataset[idx])
                for idx in np.random.choice([
                    _idx for _idx, _data in enumerate(self.few_shot_dataset)
                    if _data['name'] != input_data['name']
                ], size=self.few_shot_num, replace=False)
            ] + [input_data.get('_extra_header', str())]),
        }
    
    def sample(self, **kwargs):
        raise NotImplementedError