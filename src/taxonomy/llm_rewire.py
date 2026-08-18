import json
import os
import torch
import argparse
import graph_utils as utils
import config
import re
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
import csv
from clean import TaxonCleaner
import pickle
import numpy as np
load_dotenv(override = True)
access_token_read = os.getenv('access_token_read_hf')
login(token = access_token_read)


############################################################################
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

# Chat template (usually used by instruction-tuned models)
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


def generate_answer_by_llm(batch_messages, tokenizer, model, args):

    input_ids = tokenizer.apply_chat_template(
        batch_messages, 
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False, # disable thinking for qwen3 model
        padding=True,
    ).to(model.device)

    # settings by default
    generation_kwargs = {
        "max_new_tokens": args.max_token,
        "do_sample": True,
        # "do_sample": False,
        "temperature": args.temp,
        "repetition_penalty": 1.1,
        # "return_full_text": False,
        "output_logits": True,
        "return_dict_in_generate": True,
    }
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
    
    # special bugs for Mistral8x7B, avoid early stop for numbers
    if args.llm == 'mixtral8x7b':
        generation_kwargs["eos_token_id"] = None
        generation_kwargs["min_new_tokens"] = 100
    
    gen_output = model.generate(
        input_ids,
        # **generation_kwargs,
        generation_config=GenerationConfig(
            **generation_kwargs
        ),
    )
    
    all_sequences = gen_output.sequences
    all_gen_logits = gen_output.logits
    # reshape gen_logits tuple to (batch_size, seq_length, vocab_size)
    all_gen_logits = torch.stack(all_gen_logits, dim=0)
    # all_gen_logits = all_gen_logits.view(input_ids.shape[0], -1, all_gen_logits.shape[-1])
    all_gen_logits = all_gen_logits.transpose(0, 1) # the only change
    # gen_logits = gen_output.logits
    all_answer_ids = all_sequences[..., input_ids.shape[-1]:]
    # answer_ids = sequences[..., input_ids.shape[-1]:].squeeze(0)
    all_responses = tokenizer.batch_decode(all_answer_ids, skip_special_tokens=True)
    # response = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return all_answer_ids, all_gen_logits, all_responses



def get_tokenizer_and_model(model_name):
    # quantization configuration
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    # )
    # Specific for gemma3 models
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


def get_answer_and_prob(answer_with_think_ids, logits, answer, tokenizer):
    '''
    Process answer and its probability from logits.
    logits: usually a tuple/list (gen_len) of tensors (batch, vocab).
    '''
    ans_whole = answer
    sid, eid = extract_answer_token(answer)
    ans_token = answer[sid:eid] if sid is not None and eid is not None else answer

    # tokenize the response
    if not hasattr(tokenizer, "encode"): # gemma3processor
        ans_ids = tokenizer.tokenizer(ans_whole, add_special_tokens=False).input_ids
    else: # mistral, llama, qwen
        ans_ids = tokenizer.encode(ans_whole, add_special_tokens=False)

    if answer_with_think_ids is None or len(answer_with_think_ids) == 0:
        print("WARNING: No valid tokens found in the answer.")
        return None, None

    if logits is None:
        print("WARNING: No logits returned by generation.")
        return ans_token, None

    # gen_output.logits is typically a tuple/list of length gen_len:
    # each item is (batch, vocab). Here batch is 1.
    try:
        step_logits = torch.stack(list(logits), dim=0).squeeze(1)  # (gen_len, vocab)
    except Exception as e:
        print(f"WARNING: Failed to stack logits: {e}")
        return ans_token, None

    probs = torch.nn.functional.softmax(step_logits, dim=-1)  # (gen_len, vocab)
    token_probs = []
    for tok in ans_ids:
        try:
            exact_ans = tokenizer.decode([tok], skip_special_tokens=True)
        except Exception:
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

        if pos < 0 or pos >= probs.shape[0]:
            continue
        prob = probs[pos, tok]
        token_probs.append(prob)
        break # only consider the first token that matches the answer
    
    if len(token_probs) == 0:
        print("WARNING: No valid tokens found in the answer.")
        return None, None
    # multiple tokens for word confidence
    conf = torch.prod(torch.tensor(token_probs))
    return ans_token, conf


def get_valid_answer_with_prob(messages, model, tokenizer, args, max_retry=3):
    '''
    Get valid answer and its probability from the response.
    '''

    all_whole_answer = []
    all_answer = []
    all_answer_conf = []

    all_ans_with_think_ids, all_gen_logits, all_answer_with_thinking = generate_answer_by_llm(messages, tokenizer, model, args)
    # check validity of the answer
    for i, answer_with_thinking in enumerate(all_answer_with_thinking):
        possible_ans = ["Answer", "**False**", "**True**"]
        for ans_key in possible_ans:
            ans_pos = answer_with_thinking.rfind(ans_key)
            if ans_pos >= 0:
                break
        if ans_pos < 0:
            print(f"WARNING: Answer not found in the response. Please Redo the prompting. \nInput: {messages}. \nCurrent response: {answer_with_thinking}")
            all_whole_answer.append(answer_with_thinking)
            all_answer.append('None')
            all_answer_conf.append(0.0)
            continue
        # pre-process the answer
        res = answer_with_thinking[ans_pos:]
        sid, eid = extract_answer_token(res)
        ans_value = res[sid:eid].strip() if sid is not None and eid is not None else res
        if "true" not in ans_value.lower() and "false" not in ans_value.lower() and "yes" not in ans_value.lower() and "no" not in ans_value.lower():
            print(f"WARNING: Answer must be 'True' or 'False'. \nInput: {messages}. \nCurrent response: {answer_with_thinking}, res: {ans_value}")
            all_whole_answer.append(answer_with_thinking)
            all_answer.append('None')
            all_answer_conf.append(0.0)
            continue
        # get confidence score
        answer, answer_conf = get_answer_and_prob(all_ans_with_think_ids[i], all_gen_logits[i], res, tokenizer)
        # turn answer to true/false if we have yes/no answer
        if answer is not None:
            if "yes" in answer.lower():
                answer = "True"
            elif "no" in answer.lower():
                answer = "False"
        else:
            answer = 'None'
        if answer_conf is None:
            answer_conf = 0.0
        all_whole_answer.append(answer_with_thinking)
        all_answer.append(answer)
        all_answer_conf.append(answer_conf)
    assert len(all_whole_answer) == len(all_answer) == len(all_answer_conf)
    return all_whole_answer, all_answer, all_answer_conf


def load_submatrix(emb_pkl_path, sub_nodes):
    """
    Load a pre-computed global embedding matrix and extract the sub-matrix
    for a subset of nodes (e.g. a specific model's taxonomy).

    The pkl file is expected to contain {"nodes": [...], "emb": np.ndarray}.

    Args:
        emb_pkl_path : path to the global embedding pickle file
        sub_nodes    : list of node ids required (e.g. sorted(dag.nodes()))

    Returns:
        emb_sub : (len(sub_nodes), d) numpy array, rows aligned with sub_nodes
    """
    with open(emb_pkl_path, 'rb') as f:
        data = pickle.load(f)
    global_nodes = data["nodes"]
    global_emb = data["emb"]

    # if not global_nodes[0].startswith('wd:'): 
    #     global_nodes = ['wd:' + node for node in global_nodes]

    global_node2idx = {v: i for i, v in enumerate(global_nodes)}

    missing = [v for v in sub_nodes if v not in global_node2idx]
    if missing:
        raise KeyError(
            f"{len(missing)} nodes not found in global embeddings, "
            f"first 5: {missing[:5]}"
        )

    indices = np.array([global_node2idx[v] for v in sub_nodes])
    return global_emb[indices]


def load_rewire_results(rewire_file_loc):
    if not os.path.exists(rewire_file_loc):
        raise FileNotFoundError(f"Rewire file not found: {rewire_file_loc}")
    with open(rewire_file_loc, 'r') as f:
        rewire_res = json.load(f)
    llm_rewire_res = dict()
    for res in rewire_res:
        parent_, child_ = res['id'].split('_')
        llm_rewire_res[(child_, parent_)] = res
    return llm_rewire_res



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='******Running LLMs******')
    parser.add_argument("--temp", type=float, default=0.01, help="Temperature of LLM")
    parser.add_argument("--max_token", type=int, default=50, help="Max output token of LLM")
    parser.add_argument("--prompt", type=str, default="SubClassEval.txt", help="Prompt template for Semantic Prediction")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")

    args = parser.parse_args()
    print("Arguments:", args)
    
    # Parameters
    LLMs = config.LLM_MODELS
    LABELS_FILE = config.TAXONOMY_LABELS_FILE
    DESCRIPTIONS_FILE = config.TAXONOMY_DESCRIPTIONS_FILE
    HIERARCHY_FILE = config.TAXONOMY_FILE
    CLS_INST_COUNT_FILE = config.CLS_INST_COUNT_FILE
    OUTPUT_FOLDER = config.LLM_OUTPUT_DIR
    MAJORITY_REWIRE_LINKS_FILE = config.MAJORITY_REWIRE_LINKS_FILE
    MAJORITY_PREDICTIONS_FILE = config.MAJORITY_PREDICTIONS_FILE
    MAJORITY_PREDICTIONS_REWIRE_FILE = config.MAJORITY_PREDICTIONS_REWIRE_FILE
    PROMPTS_FOLDER = config.PROMPTS_DIR
    threshold = 0.5
    cls2label = utils.load_labels(LABELS_FILE)
    cls2desc = utils.load_descriptions(DESCRIPTIONS_FILE)

    # loading class instance count from class_instance_count.csv
    CLS_INST_COUNT = dict()
    with open(CLS_INST_COUNT_FILE, "r", encoding="utf-8") as f:
        reader = csv.reader(f) # "class", "instance_count"
        for row in reader:
            if row[0].startswith("http://www.wikidata.org/entity/"):
                cls = row[0].split("/")[-1]
                CLS_INST_COUNT[cls] = int(row[1])

    # first get the rewire links from majority predictions
    cleaner = TaxonCleaner(output_dir=OUTPUT_FOLDER, init_taxonomy=HIERARCHY_FILE, 
                                 cls_inst_count=CLS_INST_COUNT, models=LLMs)
    cleaner.get_majority_predictions(file_name=MAJORITY_PREDICTIONS_FILE, threshold=threshold)

    # calculate similarity matrix
    emb_pkl = config.EMBEDDING_PKL_FILE
    all_nodes = sorted(list(cleaner.wiki_dag.nodes()))
    if os.path.exists(emb_pkl):
        emb_sub = load_submatrix(emb_pkl, all_nodes)
        simi_matrix = emb_sub @ emb_sub.T
        node2id = {node: i for i, node in enumerate(all_nodes)}
    else:
        model_name = 'Lihuchen/pearl_small' # or 'sentence-transformers/LaBSE'
        Str_model = AutoModel.from_pretrained(model_name)
        Str_tokenizer = AutoTokenizer.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Str_model.to(device) # gpu
        print("Calculating similarity matrix...") 
        simi_matrix, node2id = utils.calculate_simi_matrix(cleaner.wiki_dag, Str_model, Str_tokenizer, cls2label, file_path=emb_pkl)
    
    # get the rewire links
    cleaner.cut(simi_matrix, node2id)
    cleaner.resolve(simi_matrix, node2id)
    cleaner.reduce()
    # rewire
    bfs_edges = utils.bfs_edges_by_level(cleaner.wiki_dag, cleaner.root)
    bfs_edges = utils.reorder_edges_by_similarity(cleaner.wiki_dag, bfs_edges, simi_matrix, node2id, reverse=True)
    cleaner.get_reiwre_links(bfs_edges)
    cleaner.store_rewire_links(os.path.join(OUTPUT_FOLDER, MAJORITY_REWIRE_LINKS_FILE)) # child -> parent format edge

    # load rewire links to hierrels
    hierrels = []
    with open(os.path.join(OUTPUT_FOLDER, MAJORITY_REWIRE_LINKS_FILE), 'r') as f:
        for line in f:
            child, parent = line.strip().split(',')
            if child not in cls2label or parent not in cls2label:
                raise ValueError(f"Child or parent not found in literals: {child}, {parent}")
            if child not in cls2desc or parent not in cls2desc:
                raise ValueError(f"Child or parent not found in descriptions: {child}, {parent}")
            hierrels.append(tuple([child, parent]))
    # load prompt template
    if os.path.exists(os.path.join(PROMPTS_FOLDER, args.prompt)):
        with open(os.path.join(PROMPTS_FOLDER, args.prompt), 'r') as f:
            prompt_tmp = f.read()
    
    ############################################################################################################################
    # Inference on rewire links
    ############################################################################################################################

    for model_name in LLMs:
        # add llm name to args
        args.llm = model_name
        # load model and rewire links
        try:
            tokenizer, model = get_tokenizer_and_model(model_name)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
    
        print("Start inference...")
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)

        batched_messages = []
        batched_messages_inverse = []
        batched_parents = []
        batched_children = []
        with open(os.path.join(OUTPUT_FOLDER, f"{model_name}_predictions_rewire.json"), 'a') as f:
            for i, (child, parent) in enumerate(tqdm(hierrels)):
                problem_text = prompt_tmp.format(
                    parent_label=cls2label[parent], parent_desc=cls2desc[parent],
                    child_label=cls2label[child], child_desc=cls2desc[child])
                problem_text_inverse = prompt_tmp.format(
                    parent_label=cls2label[child], parent_desc=cls2desc[child],
                    child_label=cls2label[parent], child_desc=cls2desc[parent])
                messages = get_prompt_text(problem_text, args)
                messages_inverse = get_prompt_text(problem_text_inverse, args)

                # batching
                batched_messages.append(messages)
                batched_messages_inverse.append(messages_inverse)
                batched_parents.append(parent)
                batched_children.append(child)
                if len(batched_messages) < args.batch_size and (i != len(hierrels) - 1):
                    continue

                # LLM inference
                print(f"Processing ids: {i+1-args.batch_size} to {i+1} ...")
                all_ans_with_thinking, all_answer, all_answer_conf = get_valid_answer_with_prob(batched_messages, model, tokenizer, args, max_retry=5)
                all_ans_with_thinking_inverse, all_answer_inverse, all_answer_conf_inverse = get_valid_answer_with_prob(batched_messages_inverse, model, tokenizer, args, max_retry=5)
                assert len(all_ans_with_thinking) == len(batched_messages)
                assert len(all_ans_with_thinking_inverse) == len(batched_messages_inverse)

                # store results
                for j in range(len(all_answer)):
                    results_ = {
                        "id": f"{batched_parents[j]}_{batched_children[j]}",
                        "plabel": cls2label[batched_parents[j]],
                        "clabel": cls2label[batched_children[j]],
                        "original_answer": all_ans_with_thinking[j],
                        "original_answer_inverse": all_ans_with_thinking_inverse[j],
                        "answer": all_answer[j].strip() if all_answer[j] is not None else None,
                        "confidence": round(float(all_answer_conf[j]), 5) if all_answer_conf[j] is not None else None,
                        "answer_inverse": all_answer_inverse[j].strip() if all_answer_inverse[j] is not None else None,
                        "confidence_inverse": round(float(all_answer_conf_inverse[j]), 5) if all_answer_conf_inverse[j] is not None else None,
                    }
                    json.dump(results_, f, ensure_ascii=False)
                    f.write('\n')
                
                # clear the batched messages
                batched_messages = []
                batched_messages_inverse = []
                batched_parents = []
                batched_children = []

        print("  Done! Results saved to", os.path.join(OUTPUT_FOLDER, f"{model_name}_predictions_rewire.json"))
    
    ############################################################################################################################
    # Merge results by majority voting
    ############################################################################################################################
    # transform the results to a single JSON file
    for model_name in LLMs:
        utils.json_transform([os.path.join(OUTPUT_FOLDER, f"{model_name}_predictions_rewire.json")], os.path.join(OUTPUT_FOLDER, f"{model_name}_predictions_rewire.json"))

    # load all json results
    results_collections = []
    id2modelname = dict()
    for i, model_name in enumerate(LLMs):
        id2modelname[i] = model_name
        results_collections.append(load_rewire_results(os.path.join(OUTPUT_FOLDER, f"{model_name}_predictions_rewire.json")))
    

    with open(os.path.join(OUTPUT_FOLDER, MAJORITY_PREDICTIONS_REWIRE_FILE), 'w') as f:
        final_results = []
        for edge in hierrels:
            child, parent = edge
            pred_counts = dict()
            pred_confs = dict()
            for i, model_res in enumerate(results_collections):
                if (child, parent) not in model_res:
                    raise ValueError(f"Edge {edge} not found in {id2modelname[i]} rewire predictions.")
                ans = model_res[(child, parent)]['answer']
                inv_ans = model_res[(child, parent)]['answer_inverse']
                conf = model_res[(child, parent)]['confidence']
                inv_conf = model_res[(child, parent)]['confidence_inverse']
                if ans.lower() == 'true' and inv_ans.lower() == 'false' and min(conf, inv_conf) >= threshold:
                    pred_counts['[SUBSUME]'] = pred_counts.get('[SUBSUME]', 0) + 1
                    pred_confs['[SUBSUME]'] = min(pred_confs.get('[SUBSUME]', 1), min(conf, inv_conf))
                elif ans.lower() == 'false' and inv_ans.lower() == 'true' and min(conf, inv_conf) >= threshold:
                    pred_counts['[REVERSE]'] = pred_counts.get('[REVERSE]', 0) + 1
                elif ans.lower() == 'true' and inv_ans.lower() == 'true' and min(conf, inv_conf) >= threshold:
                    pred_counts['[EQUIVALENT]'] = pred_counts.get('[EQUIVALENT]', 0) + 1
                elif ans.lower() == 'false' and inv_ans.lower() == 'false' and min(conf, inv_conf) >= threshold:
                    pred_counts['[IRRELEVANT]'] = pred_counts.get('[IRRELEVANT]', 0) + 1
            
            if not pred_counts:
                max_count = 0
                n_max_counts = 2
            else:
                max_count = max(pred_counts.values())
                n_max_counts = sum(1 for count in pred_counts.values() if count == max_count)
            if pred_counts.get('[SUBSUME]', 0) == max_count and n_max_counts < 2: # valid rewire
                res_ = dict()
                res_['id'] = results_collections[0][(child, parent)]['id']
                res_['plabel'] = results_collections[0][(child, parent)]['plabel']
                res_['clabel'] = results_collections[0][(child, parent)]['clabel']
                res_['answer'] = 'True'
                res_['confidence'] = pred_confs['[SUBSUME]']
                res_['answer_inverse'] = 'False'
                res_['confidence_inverse'] = pred_confs['[SUBSUME]']
                final_results.append(res_)
            else:
                res_ = dict()
                res_['id'] = results_collections[0][(child, parent)]['id']
                res_['plabel'] = results_collections[0][(child, parent)]['plabel']
                res_['clabel'] = results_collections[0][(child, parent)]['clabel']
                res_['answer'] = 'False'
                res_['confidence'] = 0.0
                res_['answer_inverse'] = 'False'
                res_['confidence_inverse'] = 0.0
                final_results.append(res_)
        json.dump(final_results, f, indent=4, ensure_ascii=False)
