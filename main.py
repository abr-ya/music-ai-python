import base64
import uuid
import modal
import os

from pydantic import BaseModel
import requests

from typing import List

from prompts import LYRICS_GENERATOR_PROMPT, PROMPT_GENERATOR_PROMPT

app = modal.App("music-generator")

image = (
  modal.Image.debian_slim()
    .apt_install("git", "ffmpeg")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step", "cd /tmp/ACE-Step && pip install ."])
    .env({"HF_HOME": "/.cache/huggingface"})
    .add_local_python_source("prompts")
)

model_volume = modal.Volume.from_name("ace-step-models", create_if_missing=True)
hf_volume = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)

music_gen_secrets = modal.Secret.from_name("music-gen-secret")

class AudioGenerationBase(BaseModel):
    audio_duration: float = 60.0
    seed: int = -1
    guidance_scale: float = 15.0
    infer_step: int = 60
    instrumental: bool = False

class GenerateFromDescriptionRequest(AudioGenerationBase):
    full_described_song: str

class GenerateWithCustomLyricsRequest(AudioGenerationBase):
    prompt: str
    lyrics: str

class GenerateWithDescribedLyricsRequest(AudioGenerationBase):
    prompt: str
    described_lyrics: str

class GenerateMusicResponseS3(BaseModel):
    s3_key: str
    cover_image_s3_key: str
    categories: List[str]

class GenerateMusicResponse(BaseModel):
    audio_data: str

@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/models": model_volume, "/.cache/huggingface": hf_volume},
    secrets=[music_gen_secrets],
    scaledown_window=15
)

class MusicGenServer:
    @modal.enter()
    def load_model(self):
        from acestep.pipeline_ace_step import ACEStepPipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from diffusers import AutoPipelineForText2Image
        import torch

        # Music Generation Model
        self.music_model = ACEStepPipeline(
            checkpoint_dir="/models",
            dtype="bfloat16",
            torch_compile=False,
            cpu_offload=False,
            overlapped_decode=False
        )

        # Large Language Model
        model_id = "Qwen/Qwen2-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir="/.cache/huggingface"
        )

        # Stable Diffusion Model (thumbnails)
        self.image_pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", cache_dir="/.cache/huggingface")
        self.image_pipe.to("cuda")

  
    def prompt_qwen(self, question: str):
      messages = [
          {"role": "user", "content": question}
      ]
      text = self.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
      )
      model_inputs = self.tokenizer(
          [text], return_tensors="pt").to(self.llm_model.device)

      generated_ids = self.llm_model.generate(
          model_inputs.input_ids,
          max_new_tokens=512
      )
      generated_ids = [
          output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
      ]

      response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

      return response

    def generate_prompt(self, description: str):
        # Insert description into template
        full_prompt = PROMPT_GENERATOR_PROMPT.format(user_prompt=description)
        # Run LLM inference and return that
        return self.prompt_qwen(full_prompt)

    def generate_lyrics(self, description: str):
        # Insert description into template
        full_prompt = LYRICS_GENERATOR_PROMPT.format(description=description)
        # Run LLM inference and return that
        return self.prompt_qwen(full_prompt)

    @modal.fastapi_endpoint(method="POST")
    def generate(self) -> GenerateMusicResponse:
        output_dir = "/tmp/outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{uuid.uuid4()}.mp3")

        self.music_model(
            prompt="Rock melody by few cellos like Smoke on the Water by Deep Purple and a drum kit, high energy, epic, cinematic",
            audio_duration=30,
            save_path=output_path,
            lyrics="[inst]",
            format="mp3",
        )

        with open(output_path, "rb") as f:
            audio_data = f.read()

        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        os.remove(output_path)

        return GenerateMusicResponse(audio_data=audio_b64)
    
    @modal.fastapi_endpoint(method="POST")
    def generate_from_description(self, request: GenerateFromDescriptionRequest) -> GenerateMusicResponseS3:
        # Generating a prompt
        prompt = self.generate_prompt(request.full_described_song)

        # Generating lyrics
        lyrics = ""
        if not request.instrumental:
            lyrics = self.generate_lyrics(request.full_described_song)
        return self.generate_and_upload_to_s3(prompt=prompt, lyrics=lyrics,
                                              description_for_categorization=request.full_described_song, **request.model_dump(exclude={"full_described_song"}))

    @modal.fastapi_endpoint(method="POST")
    def generate_with_lyrics(self, request: GenerateWithCustomLyricsRequest) -> GenerateMusicResponseS3:
        pass

    @modal.fastapi_endpoint(method="POST")
    def generate_with_described_lyrics(self, request: GenerateWithDescribedLyricsRequest) -> GenerateMusicResponseS3:
        # Generating lyrics
        lyrics = ""
        pass
    
@app.local_entrypoint()
def main():
    server = MusicGenServer()
    endpoint_url = server.generate.get_web_url()
    print(f"Music generation endpoint is available at: {endpoint_url}")

    response = requests.post(endpoint_url)
    response.raise_for_status()
    music_response = GenerateMusicResponse(**response.json())

    audio_bytes = base64.b64decode(music_response.audio_data)
    output_file = "generated_music.mp3"
    with open(output_file, "wb") as f:
        f.write(audio_bytes)
