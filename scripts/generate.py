import argparse

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests

from molmoact_device import resolve_device

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default='/weka/oe-training-default/jiafeid/hqfang/checkpoints/checkpoints-test-act/step1-hf')
parser.add_argument("--image", default="/weka/oe-training-default/jiafeid/hqfang/data/OXE/pose/bc_z/0000000/0000.png")
parser.add_argument("--device", default=None, help="for example xpu:0 or cuda:0; default is the best available backend")
parser.add_argument("--max-new-tokens", type=int, default=2048)
args = parser.parse_args()

device = str(resolve_device(args.device))

# load the processor
processor = AutoProcessor.from_pretrained(
    args.checkpoint,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map={"": device}
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    args.checkpoint,
    trust_remote_code=True,
    torch_dtype='auto',
    device_map={"": device}
)

model.to(device)

language_instruction = "pick coke can"
full_instruction = f"The task is {language_instruction}. What is the action that the robot should take. To figure out the action that the robot should take to {language_instruction}, let's think through it step by step. First, what is the depth map for this image? Second, what is the trajectory of the end effector? Based on the depth map of the image and the trajectory of the end effector, what is the action that the robot should take?"
print("***** instruction *****")
print(full_instruction)

# process the image and text

image_path = args.image
image = Image.open(image_path)

# process the image and text
inputs = processor.process(
    images=[image],
    text=full_instruction,
)

# move inputs to the correct device and make a batch of size 1
inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

# print(inputs['input_ids'])

# print(processor.tokenizer.decode(inputs['input_ids'][0]))

# generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
output = model.generate_from_batch(
    inputs,
    GenerationConfig(max_new_tokens=args.max_new_tokens, stop_strings="<|endoftext|>"),
    tokenizer=processor.tokenizer
)

# only get generated tokens; decode them to text
generated_tokens = output[0,inputs['input_ids'].size(1):]
# print(generated_tokens)
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

# print the generated text
print("***** model prediction *****")
print(generated_text)



