import logging
import subprocess
import sys
import time
from dataclasses import field
from pathlib import Path

import hydra

from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, nested_dataclass, setup_logging

LOG = logging.getLogger(__name__)


@nested_dataclass(kw_only=True)
class RivaGenerateConfig(GenerateSolutionsConfig):
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)
    
    generation_type: str = 'tts'  # 'tts' or 'asr'
    voice: str = 'Magpie-Multilingual.EN-US.Emma'
    tts_output_dir: str = '/tmp/tts_outputs'
    language_code: str = 'en-US'
    sample_rate_hz: int = 22050
    
    # ASR parameters  
    automatic_punctuation: bool = True
    speaker_diarization: bool = False
    
    generation_key: str = "result"
    prompt_config: str | None = None
    prompt_format: str = "openai"  # Bypass prompt validation for TTS/ASR


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_riva_generate_config", node=RivaGenerateConfig)


class RivaGenerationTask(GenerationTask):
    def __init__(self, cfg: RivaGenerateConfig):
        self.riva_cfg = cfg
        super().__init__(cfg)
    
    def wait_for_server(self):
        """Override to wait for gRPC server instead of HTTP."""
        host = self.cfg.server.get('host', '127.0.0.1')
        grpc_port = int(self.cfg.server.get('port', '8000')) + 1
        
        LOG.info(f"Waiting for Riva gRPC server at {host}:{grpc_port}")
        
        # Try to connect with grpc_health_probe or netcat
        max_attempts = 60  # Wait up to 3 minutes
        for attempt in range(max_attempts):
            # Check if port is open using netcat
            result = subprocess.run(
                f"nc -zv {host} {grpc_port}",
                shell=True,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                LOG.info(f"Riva gRPC server is ready at {host}:{grpc_port}")
                # Give it a bit more time to fully initialize
                time.sleep(2)
                return
            
            if attempt < max_attempts - 1:
                time.sleep(3)
        
        LOG.warning(f"Riva gRPC server not responding after {max_attempts*3} seconds")
    
    def setup_llm(self):
        host = self.cfg.server.get('host', '127.0.0.1')
        grpc_port = str(int(self.cfg.server.get('port', '8000')) + 1)
        
        if self.riva_cfg.generation_type == 'tts':
            from nemo_skills.inference.model.tts_nim import TTSNIMModel
            Path(self.riva_cfg.tts_output_dir).mkdir(parents=True, exist_ok=True)
            return TTSNIMModel(
                host=host, 
                port=grpc_port, 
                model='riva-tts',  # Required by BaseModel
                voice=self.riva_cfg.voice,
                language_code=self.riva_cfg.language_code,
                sample_rate_hz=self.riva_cfg.sample_rate_hz,
                output_dir=self.riva_cfg.tts_output_dir
            )
        else:
            from nemo_skills.inference.model.asr_nim import ASRNIMModel
            return ASRNIMModel(
                host=host, 
                port=grpc_port, 
                model='riva-asr',  # Required by BaseModel
                language_code=self.riva_cfg.language_code
            )
    
    def setup_prompt(self):
        return None
    
    def fill_prompt(self, data_point, all_data):
        if self.riva_cfg.generation_type == 'tts':
            return data_point.get('text', data_point.get('prompt', ''))
        else:
            return data_point.get('audio_path', data_point.get('audio_file', ''))
    
    def log_example_prompt(self, data):
        if data:
            LOG.info(f"Example input: {self.fill_prompt(data[0], data)}")
    
    async def process_single_datapoint(self, data_point, all_data):
        prompt = self.fill_prompt(data_point, all_data)
        if not prompt:
            return {self.cfg.generation_key: "", "error": "Empty input"}
        
        extra_body = dict(self.cfg.inference.extra_body) if self.cfg.inference.extra_body else {}
        
        if self.riva_cfg.generation_type == 'tts':
            # Pass through all TTS and zero-shot parameters from data_point
            for key in ['voice', 'language_code', 'zero_shot_audio_prompt_file', 
                        'zero_shot_transcript', 'zero_shot_quality', 'sample_rate_hz']:
                if key in data_point:
                    extra_body[key] = data_point[key]
        else:
            extra_body.update({
                'automatic_punctuation': self.riva_cfg.automatic_punctuation,
                'speaker_diarization': self.riva_cfg.speaker_diarization,
            })
        
        return await self.llm.generate_async(
            prompt=prompt,
            tokens_to_generate=1,
            temperature=0.0,
            extra_body=extra_body,
            **self.extra_generate_params
        )


GENERATION_TASK_CLASS = RivaGenerationTask


@hydra.main(version_base=None, config_name="base_riva_generate_config")
def generate(cfg: RivaGenerateConfig):
    cfg = RivaGenerateConfig(_init_nested=True, **cfg)
    task = RivaGenerationTask(cfg)
    task.generate()


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(get_help_message(RivaGenerateConfig, server_params=server_params()))
    else:
        setup_logging()
        generate()
