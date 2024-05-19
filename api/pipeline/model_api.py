import onnxruntime_genai as og
import os
from dataclasses import dataclass,asdict
from api import RagEngine


@dataclass
class GenerationConfig:
    temperature: float
    do_sample: bool
    max_length: int
    top_k: int
    top_p: float
    repetition_penalty: float



class ModelInfer:

    def __init__(self):
        self.model = None

    def chat_template_with_context(self,user_prompt,context):
        return '<|user|>\n{context} {input} Use the context above if it is helpful.<|end|>\n<|assistant|>'.format(input=user_prompt,context=context)


    def initliaze_model(self):
        if self.model is None:
            self.rag_engine = RagEngine()
            self.model = og.Model(os.path.abspath("./models/"))
            self.tokenizer = og.Tokenizer(self.model)
            self.tokenizer_stream = self.tokenizer.create_stream()

    def generate(self,generation_config: GenerationConfig,user_prompt:str):
        #prompt = f'{chat_template.format(input=user_prompt)}'
        context = self.rag_engine.create_context_processor(user_prompt)
        input_tokens = self.tokenizer.encode(self.chat_template_with_context(user_prompt,context))
        params = og.GeneratorParams(self.model)
        params.set_search_options(**asdict(generation_config))
        params.input_ids = input_tokens
        self.generator = og.Generator(self.model, params)

    def infer(self,user_prompt:str):
        self.initliaze_model()
        generation_config = GenerationConfig(
            temperature=0.0,
            do_sample=False,
            max_length=2048,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.0
        )
        self.generate(generation_config,user_prompt.strip())
        try:
            while not self.generator.is_done():
                self.generator.compute_logits()
                self.generator.generate_next_token()
                new_token = self.generator.get_next_tokens()[0]
                print(self.tokenizer_stream.decode(new_token), end='', flush=True)
        except KeyboardInterrupt:
            print("  --control+c pressed, aborting generation--")
        print()
        print()
        del self.generator


if __name__ == '__main__':
    model = ModelInfer()
    model.infer("hello")



""" model = og.Model(os.path.abspath("./models/"))
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()
prompt = '<|user|>\nhello <|end|>\n<|assistant|>'

input_tokens = tokenizer.encode(prompt)
params = og.GeneratorParams(model)
params.input_ids = input_tokens
generator = og.Generator(model, params)
print()
print("Output: ", end='', flush=True)
try:
    while not generator.is_done():
        generator.compute_logits()
        generator.generate_next_token()
        new_token = generator.get_next_tokens()[0]
        print(tokenizer_stream.decode(new_token), end='', flush=True)
except KeyboardInterrupt:
    print("  --control+c pressed, aborting generation--")
print()
print() """