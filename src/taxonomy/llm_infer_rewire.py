import json
import os
import torch
import argparse
import json
import utils
import re
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
import pandas as pd
load_dotenv(override = True)
access_token_read = os.getenv('access_token_read_hf')
login(token = access_token_read)



llms = {
    "gemma4b": "google/gemma-3-4b-it",
    "gemma12b": "google/gemma-3-12b-it",
    "gemma27b": "google/gemma-3-27b-it",
    "qwen8b": "Qwen/Qwen3-8B",
    "qwen14b": "Qwen/Qwen3-14B",
    "qwen32b": "Qwen/Qwen3-32B",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral24b": "mistralai/Mistral-Small-24B-Instruct-2501",
    "mixtral8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama70b": "meta-llama/Llama-3.3-70B-Instruct",
    # apis
    "deepseek": None,
    "gpt": None,
}

############################################################################

def get_prompt_text(problem_text, args):
    sys_prompt = "You are an ontological expert in hierarchical concept analysis."
    if args.llm in ['llama8b', 'llama70b']:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": problem_text}
        ]
    elif args.llm in ['gemma4b', 'gemma12b', 'gemma27b']:
        messages = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": sys_prompt}]
             },
            {
                "role": "user", 
                "content": [{"type": "text", "text": problem_text}]
            }
        ]
    else:
        messages = [
            {"role": "user", "content": problem_text}
        ]
    return messages



# Chat template (usually used by instruction-tuned models)
def generate_answer_by_llm(messages, tokenizer, model, args):

    input_ids = tokenizer.apply_chat_template(
        messages, 
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)

    # settings by default
    # When temperature is very low (<0.1), sampling with softmax can overflow
    # in bfloat16, producing inf/nan probabilities and crashing CUDA.
    # use_sampling = args.temp >= 0.1
    generation_kwargs = {
        "max_new_tokens": args.max_token,
        "do_sample": True,
        # "do_sample": False,
        "temperature": args.temp,
        "repetition_penalty": 1.1,
        "output_logits": True, 
        "return_dict_in_generate": True,
    }
    # if use_sampling:
    #     generation_kwargs["temperature"] = args.temp
    #     generation_kwargs["top_p"] = 0.9
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
    
    # # special bugs for Mistral8x7B, avoid early stop for numbers
    # if args.llm == 'mixtral8x7b':
    #     generation_kwargs["eos_token_id"] = None
    #     generation_kwargs["min_new_tokens"] = 100
    max_retry = 2
    success = False
    for attempt in range(max_retry):
        try:
            gen_output = model.generate(
                input_ids,
                # **generation_kwargs,
                generation_config=GenerationConfig(
                    **generation_kwargs
                ),
            )
            logits = torch.cat(gen_output.logits, dim=0)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                raise ValueError("Generation output contains NaN or Inf values.")
            # if successful, break the loop
            success = True
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}. [Retry]")
            torch.cuda.empty_cache()
    
    if not success:
        return None, None, "Error"
    sequences = gen_output.sequences
    gen_logits = gen_output.logits
    answer_ids = sequences[..., input_ids.shape[-1]:].squeeze(0)
    response = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return answer_ids, gen_logits, response



def get_tokenizer_and_model(model_name):
    # quantization configuration
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    # )
    if model_name in ['gemma4b', 'gemma12b', 'gemma27b']:
        torch._dynamo.disable() # avoid torch._dynamo.exc.RecompileLimitExceeded
        model = Gemma3ForConditionalGeneration.from_pretrained(
            llms[model_name],
            torch_dtype=torch.bfloat16,
            # quantization_config=bnb_config,
            device_map='auto'
        ).eval()
        tokenizer = AutoProcessor.from_pretrained(llms[model_name],
                                                  padding_side="left", 
                                                  use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(llms[model_name], padding_side="left", use_fast=True)
        if tokenizer.pad_token is None: # Most LLMs don't have a pad token by default
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            llms[model_name],
            torch_dtype=torch.bfloat16,
            # quantization_config=bnb_config,
            device_map='auto'
            # # llama70b: open too many files at the same time
            # low_cpu_mem_usage=True, 
            # use_safetensors=False,
        )
    model.eval()
    return tokenizer, model


def extract_answer_token(text):
    # match the first 'true/false'
    pattern1 = r'(?P<token>\s*\b(?:true|false)\b\s*)' # true/false
    pattern2 = r'(?P<token>\s*\b(?:yes|no)\b\s*)' # yes/no violation by mistral8x7b
    match = re.search(pattern1, text, re.IGNORECASE)
    match2 = re.search(pattern2, text, re.IGNORECASE)
    if match:
        # token = match.group('token')
        start = match.start('token')
        end = match.end('token')
        return start, end
    elif match2:
        start = match2.start('token')
        end = match2.end('token')
        return start, end
    else:
        return None, None

##############################################################################

def get_answer_and_prob(answer_with_think_ids, logits, answer, tokenizer):
    '''
    Process answer and its probability from logits.
    logits: (batch_size, seq_length, vocab_size)
    '''
    ans_whole = answer
    sid, eid = extract_answer_token(answer)
    ans_token = answer[sid:eid] if sid is not None and eid is not None else answer

    # tokenize the response
    if not hasattr(tokenizer, "encode"): # gemma3processor
        ans_ids = tokenizer.tokenizer(ans_whole, add_special_tokens=False).input_ids
    else: # mistral, llama, qwen
        ans_ids = tokenizer.encode(ans_whole, add_special_tokens=False)
    # token probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    token_probs = []
    for tok in ans_ids:
        exact_ans = tokenizer.decode(tok, skip_special_tokens=True)
        if exact_ans.strip() == ans_token.strip():
            pos = answer_with_think_ids.tolist().index(tok)
        elif 'true' in exact_ans.lower():
            pos = answer_with_think_ids.tolist().index(tok)
        elif 'false' in exact_ans.lower():
            pos = answer_with_think_ids.tolist().index(tok)
        elif 'yes' in exact_ans.lower():
            pos = answer_with_think_ids.tolist().index(tok)
        elif 'no' in exact_ans.lower():
            pos = answer_with_think_ids.tolist().index(tok)
        else:
            continue

        prob = probs[0, pos, tok]
        token_probs.append(prob)
        break # only consider the first token that matches the answer
    
    if len(token_probs) == 0:
        print("WARNING: No valid tokens found in the answer.")
        return None, None
    # multiple tokens for word confidence
    conf = torch.prod(torch.tensor(token_probs))
    return ans_token, conf


def get_valid_answer_with_prob(messages, model, tokenizer, args, max_retry=3):
    cnt_ans = 0
    valid_ans = False
    while not valid_ans:
        if cnt_ans >= max_retry:
            print("Already tried max retries, but still no valid answer found.")
            return answer_with_thinking, 'None', 0.0
        ans_with_think_ids, gen_logits, answer_with_thinking = generate_answer_by_llm(messages, tokenizer, model, args)

        # check validity of the answer
        possible_ans = ["Answer", "**False**", "**True**"]
        for ans_key in possible_ans:
            ans_pos = answer_with_thinking.rfind(ans_key)
            if ans_pos >= 0:
                break
        if ans_pos < 0:
            print(f"WARNING: Answer not found in the response. Please Redo the prompting. \nInput: {messages}. \nCurrent response: {answer_with_thinking}")
            cnt_ans += 1
            continue
        res = answer_with_thinking[ans_pos:]
        sid, eid = extract_answer_token(res)
        ans_value = res[sid:eid].strip() if sid is not None and eid is not None else res
        if "true" not in ans_value.lower() and "false" not in ans_value.lower() and "yes" not in ans_value.lower() and "no" not in ans_value.lower():
            print(f"WARNING: Answer must be 'True' or 'False'. Please Redo the prompting. \nInput: {messages}. \nCurrent response: {answer_with_thinking}, res: {ans_value}")
            cnt_ans += 1
            continue
        # get final valid answer with confidence score
        valid_ans = True
        # get confidence score
        logits = torch.stack(gen_logits, dim=0).view(1, len(gen_logits), -1)
        answer, answer_conf = get_answer_and_prob(ans_with_think_ids, logits, res, tokenizer)
    return answer_with_thinking, answer, answer_conf



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='******Running LLMs******')
    parser.add_argument("--llm", type=str, default="mistral7b", help="LLM used for inference")
    parser.add_argument("--temp", type=float, default=0.01, help="Temperature of LLM")
    parser.add_argument("--max_token", type=int, default=50, help="Max output token of LLM")
    parser.add_argument("--output_dir", type=str, default="../results/rewire/outputs/", help="Output directory")
    parser.add_argument("--prompt", type=str, default="SubClassEval.txt", help="Prompt template for Semantic Prediction")
    # Optional
    parser.add_argument("--start_id", type=int, default=0, help="Start id for inference")
    # parser.add_argument('--slice', type=int, default=-1, help='Slice of the taxonomy to use (0 for all)')
    parser.add_argument('--end_id', type=int, default=-1, help='End id for inference')

    args = parser.parse_args()
    print("Arguments:", args)
    # load dataset
    try:
        tokenizer, model = get_tokenizer_and_model(args.llm)
    except Exception as e:
        print(f"Error loading model {args.llm}: {e}")
    
    # File Paths
    df_labels = pd.read_csv('../../data/clean/wikidata_2026_class_labels_full.csv')
    df_descriptions = pd.read_csv('../../data/clean/wikidata_2026_class_descriptions_full.csv')
    cls2label = df_labels.set_index('item')['itemLabel'].to_dict()
    cls2desc = df_descriptions.set_index('item')['itemDesc'].to_dict()
    cls2label = {f'wd:{k}': v for k, v in cls2label.items()}
    cls2desc = {f'wd:{k}': v for k, v in cls2desc.items()}
    del df_labels, df_descriptions

    # Load undefined pairs
    hierrels = []
    with open(f'../../results/wikc_in_llms/majority/unpredicted_edges.txt', 'r') as f:
    # with open(f'../results/rewire/data/{args.llm}_rewire_links.txt', 'r') as f:
        for line in f:
            child, parent = line.strip().split('\t')
            if child not in cls2label or parent not in cls2label:
                raise ValueError(f"Child or parent not found in literals: {child}, {parent}")
            hierrels.append(tuple([child, parent]))
    if os.path.exists(os.path.join('../../prompts', args.prompt)):
        with open(os.path.join('../../prompts', args.prompt), 'r') as f:
            prompt_tmp = f.read()
    
    
    print("Start inference...")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # outputs = []
    print(len(hierrels))
    with open(os.path.join(args.output_dir, f"{args.llm}_rewire_outputs_rewire_new_large_{args.start_id}_{args.end_id}.json"), 'a') as f:
        end = args.end_id if args.end_id != -1 else None
        for i, (child, parent) in enumerate(tqdm(hierrels[args.start_id:end])):
            print(f"Processing id: {i+args.start_id}, {parent} -> {child} ...")
            problem_text = prompt_tmp.format(
                parent_label=cls2label[parent], parent_desc=cls2desc[parent],
                child_label=cls2label[child], child_desc=cls2desc[child])
            problem_text_inverse = prompt_tmp.format(
                parent_label=cls2label[child], parent_desc=cls2desc[child],
                child_label=cls2label[parent], child_desc=cls2desc[parent])
            messages = get_prompt_text(problem_text, args)
            messages_inverse = get_prompt_text(problem_text_inverse, args)

            # LLM inference
            ans_with_thinking, answer, answer_conf = get_valid_answer_with_prob(messages, model, tokenizer, args, max_retry=5)
            ans_with_thinking_inverse, answer_inverse, answer_conf_inverse = get_valid_answer_with_prob(messages_inverse, model, tokenizer, args, max_retry=5)
            
            # store results
            results_ = {
                "id": f"{parent}_{child}",
                "plabel": cls2label[parent],
                "clabel": cls2label[child],
                "original_answer": ans_with_thinking,
                "original_answer_inverse": ans_with_thinking_inverse,
                "answer": answer.strip(),
                "confidence": round(float(answer_conf), 5),
                "answer_inverse": answer_inverse.strip(),
                "confidence_inverse": round(float(answer_conf_inverse), 5),

            }
            json.dump(results_, f, ensure_ascii=False)
            f.write('\n')
            # f.flush()

        # save the outputs
        # with open(os.path.join(args.output_dir, f"{args.llm}_outputs_{args.slice}.json"), 'w') as f:
        #     json.dump(outputs, f, indent=4)
    print("  Done! Results saved to", os.path.join(args.output_dir, f"{args.llm}_rewire_outputs_{args.start_id}_{args.end_id}.json"))

