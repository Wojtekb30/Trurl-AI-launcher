print("Wczytywanie...")
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
sciezka = "plikimodelu"
tokenizer = LlamaTokenizer.from_pretrained(sciezka)
model = LlamaForCausalLM.from_pretrained(sciezka, offload_state_dict = True, device_map='auto', offload_folder="offloadcache2", torch_dtype=torch.bfloat16)
tokeny = int(input("Max nowe tokeny (domyslnie 60): "))

rola = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."

if (str(input("Type 1 to set an role: ")).strip()=="1"):
    rola = str(input("Rola: ")).strip()

pytanie = str(input("Pytanie: ")).strip()

prompt = """
<s>[INST] <<SYS>>  {} <</SYS>>

{} [/INST]

""".format(rola, pytanie)


tokenized_prompt = tokenizer(prompt, return_tensors="pt")
print("-------------")
print("-------------")
print("Przetwarzanie...")
model.eval()
with torch.no_grad():
   ostatecznywynik = tokenizer.decode(
   model.generate(**tokenized_prompt, max_new_tokens=tokeny)[0],
   skip_special_tokens=True)
   print("Bezposrednia odpowiedz:")
   print(ostatecznywynik)
   print(" ")
   print("-------------")
   print("Zadane pytanie: "+pytanie)
   print("-------------")
   exit()
exit()