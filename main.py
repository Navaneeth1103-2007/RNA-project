from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.lora.request import LoRARequest

model_id = "ibm-granite/granite-speech-3.3-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def get_prompt(question: str, has_audio: bool):
    """Build the input prompt to send to vLLM."""
    if has_audio:
        question = f"<|audio|>{question}"
    chat = [
        {
            "role": "user",
            "content": question
        }
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False)

# NOTE - you may see warnings about multimodal lora layers being ignored;
# this is okay as the lora in this model is only applied to the LLM.
model = LLM(
    model=model_id,
    enable_lora=True,
    max_lora_rank=64,
    max_model_len=2048, # This may be needed for lower resource devices.
    limit_mm_per_prompt={"audio": 1},
)

### 1. Example with Audio [make sure to use the lora]
question = "can you transcribe the speech into a written format?"
prompt_with_audio = get_prompt(
    question=question,
    has_audio=True,
)
audio = AudioAsset("mary_had_lamb").audio_and_sample_rate

inputs = {
    "prompt": prompt_with_audio,
    "multi_modal_data": {
        "audio": audio,
    }
}

outputs = model.generate(
    inputs,
    sampling_params=SamplingParams(
        temperature=0.2,
        max_tokens=64,
    ),
    lora_request=[LoRARequest("speech", 1, model_id)]
)
print(f"Audio Example - Question: {question}")
print(f"Generated text: {outputs[0].outputs[0].text}")


### 2. Example without Audio [do NOT use the lora]
question = "What is the capital of Brazil?"
prompt = get_prompt(
    question=question,
    has_audio=False,
)

outputs = model.generate(
    {"prompt": prompt},
    sampling_params=SamplingParams(
        temperature=0.2,
        max_tokens=12,
    ),
)
print(f"Text Only Example - Question: {question}")
print(f"Generated text: {outputs[0].outputs[0].text}")
