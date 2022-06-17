from transformers import GPT2Tokenizer, OPTForCausalLM
import os
#this doesn't work, because of course the primary software for a multi-billion dollar industry is stupid
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
model = OPTForCausalLM.from_pretrained("/opt/13b/metaseq/projects/OPT/models/13b")
tokenizer = GPT2Tokenizer.from_pretrained("/opt/13b/metaseq/projects/OPT/models/13b")
system('clear')
prompt = "AI: I am an artificial intelligence.\nHuman: What do you do?\nAI: I will respond to your questions, answering them to the best of my ability.\nHuman: Excellent, what is the capitol of France?\nAI: The capitol of France is Paris."
print(prompt)
new = ""
while new.lower() != 'q' or 'quit':
  new = input("Human: ")
  prompt = prompt + '\nHuman: ' + new + '\nAI: '
  if prompt.lower() == 'q' or 'quit':
      break
  inputs = tokenizer(prompt, return_tensors="pt")
  generated_id = model.generate(inputs.input_ids)
  new = "AI: " + tokenizer.batch_decode(generated_ids,skip_special_tokens=True, chean__up_tokenization_spaces=False)[0]
  print(new)
  prompt = prompt + "\n" + new
