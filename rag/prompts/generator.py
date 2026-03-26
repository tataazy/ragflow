#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import asyncio
import datetime
import json
import logging
import re
from copy import deepcopy
from typing import Tuple
import jinja2
import json_repair
from common.misc_utils import hash_str2int
from rag.nlp import rag_tokenizer
from rag.prompts.template import load_prompt
from common.constants import TAG_FLD
from common.token_utils import encoder, num_tokens_from_string

STOP_TOKEN = "<|STOP|>"
COMPLETE_TASK = "complete_task"
INPUT_UTILIZATION = 0.5


def get_value(d, k1, k2):
    return d.get(k1, d.get(k2))


def chunks_format(reference):
    if not reference or not isinstance(reference, dict):
        return []
    return [
        {
            "id": get_value(chunk, "chunk_id", "id"),
            "content": get_value(chunk, "content", "content_with_weight"),
            "document_id": get_value(chunk, "doc_id", "document_id"),
            "document_name": get_value(chunk, "docnm_kwd", "document_name"),
            "dataset_id": get_value(chunk, "kb_id", "dataset_id"),
            "image_id": get_value(chunk, "image_id", "img_id"),
            "positions": get_value(chunk, "positions", "position_int"),
            "url": chunk.get("url"),
            "similarity": chunk.get("similarity"),
            "vector_similarity": chunk.get("vector_similarity"),
            "term_similarity": chunk.get("term_similarity"),
            "doc_type": get_value(chunk, "doc_type_kwd", "doc_type"),
        }
        for chunk in reference.get("chunks", [])
    ]


def message_fit_in(msg, max_length=4000):
    def count():
        nonlocal msg
        tks_cnts = []
        for m in msg:
            tks_cnts.append({"role": m["role"], "count": num_tokens_from_string(m["content"])})
        total = 0
        for m in tks_cnts:
            total += m["count"]
        return total

    c = count()
    if c < max_length:
        return c, msg

    msg_ = [m for m in msg if m["role"] == "system"]
    if len(msg) > 1:
        msg_.append(msg[-1])
    msg = msg_
    c = count()
    if c < max_length:
        return c, msg

    ll = num_tokens_from_string(msg_[0]["content"])
    ll2 = num_tokens_from_string(msg_[-1]["content"])
    if ll / (ll + ll2) > 0.8:
        m = msg_[0]["content"]
        m = encoder.decode(encoder.encode(m)[: max_length - ll2])
        msg[0]["content"] = m
        return max_length, msg

    m = msg_[-1]["content"]
    m = encoder.decode(encoder.encode(m)[: max_length - ll2])
    msg[-1]["content"] = m
    return max_length, msg


def kb_prompt(kbinfos, max_tokens, hash_id=False):
    from api.db.services.document_service import DocumentService

    knowledges = [get_value(ck, "content", "content_with_weight") for ck in kbinfos["chunks"]]
    kwlg_len = len(knowledges)
    used_token_count = 0
    chunks_num = 0
    for i, c in enumerate(knowledges):
        if not c:
            continue
        used_token_count += num_tokens_from_string(c)
        chunks_num += 1
        if max_tokens * 0.97 < used_token_count:
            knowledges = knowledges[:i]
            logging.warning(f"Not all the retrieval into prompt: {len(knowledges)}/{kwlg_len}")
            break

    docs = DocumentService.get_by_ids([get_value(ck, "doc_id", "document_id") for ck in kbinfos["chunks"][:chunks_num]])
    docs = {d.id: d.meta_fields for d in docs}

    def draw_node(k, line):
        if line is not None and not isinstance(line, str):
            line = str(line)
        if not line:
            return ""
        return f"\n├── {k}: " + re.sub(r"\n+", " ", line, flags=re.DOTALL)

    knowledges = []
    for i, ck in enumerate(kbinfos["chunks"][:chunks_num]):
        cnt = "\nID: {}".format(i if not hash_id else hash_str2int(get_value(ck, "id", "chunk_id"), 500))
        cnt += draw_node("Title", get_value(ck, "docnm_kwd", "document_name"))
        cnt += draw_node("URL", ck['url']) if "url" in ck else ""
        for k, v in docs.get(get_value(ck, "doc_id", "document_id"), {}).items():
            cnt += draw_node(k, v)
        cnt += "\n└── Content:\n"
        cnt += get_value(ck, "content", "content_with_weight")
        knowledges.append(cnt)

    return knowledges


def memory_prompt(message_list, max_tokens):
    used_token_count = 0
    content_list = []
    for message in message_list:
        current_content_tokens = num_tokens_from_string(message["content"])
        if used_token_count + current_content_tokens > max_tokens * 0.97:
            logging.warning(f"Not all the retrieval into prompt: {len(content_list)}/{len(message_list)}")
            break
        content_list.append(message["content"])
        used_token_count += current_content_tokens
    return content_list


CITATION_PROMPT_TEMPLATE = load_prompt("citation_prompt")
CITATION_PLUS_TEMPLATE = load_prompt("citation_plus")
CONTENT_TAGGING_PROMPT_TEMPLATE = load_prompt("content_tagging_prompt")
CROSS_LANGUAGES_SYS_PROMPT_TEMPLATE = load_prompt("cross_languages_sys_prompt")
CROSS_LANGUAGES_USER_PROMPT_TEMPLATE = load_prompt("cross_languages_user_prompt")
FULL_QUESTION_PROMPT_TEMPLATE = load_prompt("full_question_prompt")
KEYWORD_PROMPT_TEMPLATE = load_prompt("keyword_prompt")
QUESTION_PROMPT_TEMPLATE = load_prompt("question_prompt")
VISION_LLM_DESCRIBE_PROMPT = load_prompt("vision_llm_describe_prompt")
VISION_LLM_FIGURE_DESCRIBE_PROMPT = load_prompt("vision_llm_figure_describe_prompt")
VISION_LLM_FIGURE_DESCRIBE_PROMPT_WITH_CONTEXT = load_prompt("vision_llm_figure_describe_prompt_with_context")
STRUCTURED_OUTPUT_PROMPT = load_prompt("structured_output_prompt")

ANALYZE_TASK_SYSTEM = load_prompt("analyze_task_system")
ANALYZE_TASK_USER = load_prompt("analyze_task_user")
NEXT_STEP = load_prompt("next_step")
REFLECT = load_prompt("reflect")
SUMMARY4MEMORY = load_prompt("summary4memory")
RANK_MEMORY = load_prompt("rank_memory")
META_FILTER = load_prompt("meta_filter")
ASK_SUMMARY = load_prompt("ask_summary")

PROMPT_JINJA_ENV = jinja2.Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True)


def citation_prompt(user_defined_prompts: dict = {}) -> str:
    template = PROMPT_JINJA_ENV.from_string(user_defined_prompts.get("citation_guidelines", CITATION_PROMPT_TEMPLATE))
    return template.render()


def citation_plus(sources: str) -> str:
    template = PROMPT_JINJA_ENV.from_string(CITATION_PLUS_TEMPLATE)
    return template.render(example=citation_prompt(), sources=sources)


async def keyword_extraction(chat_mdl, content, topn=3):
    template = PROMPT_JINJA_ENV.from_string(KEYWORD_PROMPT_TEMPLATE)
    rendered_prompt = template.render(content=content, topn=topn)

    msg = [{"role": "system", "content": rendered_prompt}, {"role": "user", "content": "Output: "}]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = await chat_mdl.async_chat(rendered_prompt, msg[1:], {"temperature": 0.2})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"^.*</think>", "", kwd, flags=re.DOTALL)
    if kwd.find("**ERROR**") >= 0:
        return ""
    return kwd


async def question_proposal(chat_mdl, content, topn=3):
    template = PROMPT_JINJA_ENV.from_string(QUESTION_PROMPT_TEMPLATE)
    rendered_prompt = template.render(content=content, topn=topn)

    msg = [{"role": "system", "content": rendered_prompt}, {"role": "user", "content": "Output: "}]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = await chat_mdl.async_chat(rendered_prompt, msg[1:], {"temperature": 0.2})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"^.*</think>", "", kwd, flags=re.DOTALL)
    if kwd.find("**ERROR**") >= 0:
        return ""
    return kwd


async def full_question(tenant_id=None, llm_id=None, messages=[], language=None, chat_mdl=None):
    from common.constants import LLMType
    from api.db.services.llm_service import LLMBundle
    from api.db.services.tenant_llm_service import TenantLLMService

    if not chat_mdl:
        if TenantLLMService.llm_id2llm_type(llm_id) == "image2text":
            chat_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_id)
        else:
            chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_id)
    conv = []
    for m in messages:
        if m["role"] not in ["user", "assistant"]:
            continue
        conv.append("{}: {}".format(m["role"].upper(), m["content"]))
    conversation = "\n".join(conv)
    today = datetime.date.today().isoformat()
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()

    template = PROMPT_JINJA_ENV.from_string(FULL_QUESTION_PROMPT_TEMPLATE)
    rendered_prompt = template.render(
        today=today,
        yesterday=yesterday,
        tomorrow=tomorrow,
        conversation=conversation,
        language=language,
    )

    ans = await chat_mdl.async_chat(rendered_prompt, [{"role": "user", "content": "Output: "}])
    ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
    return ans if ans.find("**ERROR**") < 0 else messages[-1]["content"]


async def cross_languages(tenant_id, llm_id, query, languages=[]):
    from common.constants import LLMType
    from api.db.services.llm_service import LLMBundle
    from api.db.services.tenant_llm_service import TenantLLMService

    if llm_id and TenantLLMService.llm_id2llm_type(llm_id) == "image2text":
        chat_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_id)
    else:
        chat_mdl = LLMBundle(tenant_id, LLMType.CHAT, llm_id)

    rendered_sys_prompt = PROMPT_JINJA_ENV.from_string(CROSS_LANGUAGES_SYS_PROMPT_TEMPLATE).render()
    rendered_user_prompt = PROMPT_JINJA_ENV.from_string(CROSS_LANGUAGES_USER_PROMPT_TEMPLATE).render(query=query,
                                                                                                     languages=languages)

    ans = await chat_mdl.async_chat(rendered_sys_prompt, [{"role": "user", "content": rendered_user_prompt}],
                                    {"temperature": 0.2})
    ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
    if ans.find("**ERROR**") >= 0:
        return query
    return "\n".join([a for a in re.sub(r"(^Output:|\n+)", "", ans, flags=re.DOTALL).split("===") if a.strip()])


async def content_tagging(chat_mdl, content, all_tags, examples, topn=3):
    template = PROMPT_JINJA_ENV.from_string(CONTENT_TAGGING_PROMPT_TEMPLATE)

    for ex in examples:
        ex["tags_json"] = json.dumps(ex[TAG_FLD], indent=2, ensure_ascii=False)

    rendered_prompt = template.render(
        topn=topn,
        all_tags=all_tags,
        examples=examples,
        content=content,
    )

    msg = [{"role": "system", "content": rendered_prompt}, {"role": "user", "content": "Output: "}]
    _, msg = message_fit_in(msg, chat_mdl.max_length)
    kwd = await chat_mdl.async_chat(rendered_prompt, msg[1:], {"temperature": 0.5})
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"^.*</think>", "", kwd, flags=re.DOTALL)
    if kwd.find("**ERROR**") >= 0:
        raise Exception(kwd)

    try:
        obj = json_repair.loads(kwd)
    except json_repair.JSONDecodeError:
        try:
            result = kwd.replace(rendered_prompt[:-1], "").replace("user", "").replace("model", "").strip()
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            obj = json_repair.loads(result)
        except Exception as e:
            logging.exception(f"JSON parsing error: {result} -> {e}")
            raise e
    res = {}
    for k, v in obj.items():
        try:
            if int(v) > 0:
                res[str(k)] = int(v)
        except Exception:
            pass
    return res


def vision_llm_describe_prompt(page=None) -> str:
    template = PROMPT_JINJA_ENV.from_string(VISION_LLM_DESCRIBE_PROMPT)

    return template.render(page=page)


def vision_llm_figure_describe_prompt() -> str:
    template = PROMPT_JINJA_ENV.from_string(VISION_LLM_FIGURE_DESCRIBE_PROMPT)
    return template.render()


def vision_llm_figure_describe_prompt_with_context(context_above: str, context_below: str) -> str:
    template = PROMPT_JINJA_ENV.from_string(VISION_LLM_FIGURE_DESCRIBE_PROMPT_WITH_CONTEXT)
    return template.render(context_above=context_above, context_below=context_below)


def tool_schema(tools_description: list[dict], complete_task=False):
    if not tools_description:
        return ""
    desc = {}
    if complete_task:
        desc[COMPLETE_TASK] = {
            "type": "function",
            "function": {
                "name": COMPLETE_TASK,
                "description": "When you have the final answer and are ready to complete the task, call this function with your answer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "description": "The final answer to the user's question"}},
                    "required": ["answer"]
                }
            }
        }
    for idx, tool in enumerate(tools_description):
        name = tool["function"]["name"]
        desc[name] = tool

    return "\n\n".join([f"## {i + 1}. {fnm}\n{json.dumps(des, ensure_ascii=False, indent=4)}" for i, (fnm, des) in
                        enumerate(desc.items())])


def form_history(history, limit=-6):
    context = ""
    for h in history[limit:]:
        if h["role"] == "system":
            continue
        role = "USER"
        if h["role"].upper() != role:
            role = "AGENT"
        context += f"\n{role}: {h['content'][:2048] + ('...' if len(h['content']) > 2048 else '')}"
    return context


async def analyze_task_async(chat_mdl, prompt, task_name, tools_description: list[dict],
                             user_defined_prompts: dict = {}):
    tools_desc = tool_schema(tools_description)
    context = ""

    if user_defined_prompts.get("task_analysis"):
        template = PROMPT_JINJA_ENV.from_string(user_defined_prompts["task_analysis"])
    else:
        template = PROMPT_JINJA_ENV.from_string(ANALYZE_TASK_SYSTEM + "\n\n" + ANALYZE_TASK_USER)
    context = template.render(task=task_name, context=context, agent_prompt=prompt, tools_desc=tools_desc)
    kwd = await chat_mdl.async_chat(context, [{"role": "user", "content": "Please analyze it."}])
    if isinstance(kwd, tuple):
        kwd = kwd[0]
    kwd = re.sub(r"^.*</think>", "", kwd, flags=re.DOTALL)
    if kwd.find("**ERROR**") >= 0:
        return ""
    return kwd


async def next_step_async(chat_mdl, history: list, tools_description: list[dict], task_desc,
                          user_defined_prompts: dict = {}):
    if not tools_description:
        return "", 0
    desc = tool_schema(tools_description)
    template = PROMPT_JINJA_ENV.from_string(user_defined_prompts.get("plan_generation", NEXT_STEP))
    user_prompt = "\nWhat's the next tool to call? If ready OR IMPOSSIBLE TO BE READY, then call `complete_task`."
    hist = deepcopy(history)
    if hist[-1]["role"] == "user":
        hist[-1]["content"] += user_prompt
    else:
        hist.append({"role": "user", "content": user_prompt})
    json_str = await chat_mdl.async_chat(
        template.render(task_analysis=task_desc, desc=desc, today=datetime.datetime.now().strftime("%Y-%m-%d")),
        hist[1:],
        stop=["<|stop|>"],
    )
    tk_cnt = num_tokens_from_string(json_str)
    json_str = re.sub(r"^.*</think>", "", json_str, flags=re.DOTALL)
    return json_str, tk_cnt


async def reflect_async(chat_mdl, history: list[dict], tool_call_res: list[Tuple], user_defined_prompts: dict = {}):
    tool_calls = [{"name": p[0], "result": p[1]} for p in tool_call_res]
    goal = history[1]["content"]
    template = PROMPT_JINJA_ENV.from_string(user_defined_prompts.get("reflection", REFLECT))
    user_prompt = template.render(goal=goal, tool_calls=tool_calls)
    hist = deepcopy(history)
    if hist[-1]["role"] == "user":
        hist[-1]["content"] += user_prompt
    else:
        hist.append({"role": "user", "content": user_prompt})
    _, msg = message_fit_in(hist, chat_mdl.max_length)
    ans = await chat_mdl.async_chat(msg[0]["content"], msg[1:])
    ans = re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)
    return """
**Observation**
{}

**Reflection**
{}
    """.format(json.dumps(tool_calls, ensure_ascii=False, indent=2), ans)


def form_message(system_prompt, user_prompt):
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def structured_output_prompt(schema=None) -> str:
    template = PROMPT_JINJA_ENV.from_string(STRUCTURED_OUTPUT_PROMPT)
    return template.render(schema=schema)


async def tool_call_summary(chat_mdl, name: str, params: dict, result: str, user_defined_prompts: dict = {}) -> str:
    template = PROMPT_JINJA_ENV.from_string(SUMMARY4MEMORY)
    system_prompt = template.render(name=name,
                                    params=json.dumps(params, ensure_ascii=False, indent=2),
                                    result=result)
    user_prompt = "→ Summary: "
    _, msg = message_fit_in(form_message(system_prompt, user_prompt), chat_mdl.max_length)
    ans = await chat_mdl.async_chat(msg[0]["content"], msg[1:])
    return re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)


async def rank_memories_async(chat_mdl, goal: str, sub_goal: str, tool_call_summaries: list[str],
                              user_defined_prompts: dict = {}):
    template = PROMPT_JINJA_ENV.from_string(RANK_MEMORY)
    system_prompt = template.render(goal=goal, sub_goal=sub_goal,
                                    results=[{"i": i, "content": s} for i, s in enumerate(tool_call_summaries)])
    user_prompt = " → rank: "
    _, msg = message_fit_in(form_message(system_prompt, user_prompt), chat_mdl.max_length)
    ans = await chat_mdl.async_chat(msg[0]["content"], msg[1:], stop="<|stop|>")
    return re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)


async def gen_meta_filter(chat_mdl, meta_data: dict, query: str) -> dict:
    meta_data_structure = {}
    for key, values in meta_data.items():
        meta_data_structure[key] = list(values.keys()) if isinstance(values, dict) else values

    sys_prompt = PROMPT_JINJA_ENV.from_string(META_FILTER).render(
        current_date=datetime.datetime.today().strftime('%Y-%m-%d'),
        metadata_keys=json.dumps(meta_data_structure),
        user_question=query
    )
    user_prompt = "Generate filters:"
    ans = await chat_mdl.async_chat(sys_prompt, [{"role": "user", "content": user_prompt}])
    ans = re.sub(r"(^.*</think>|```json\n|```\n*$)", "", ans, flags=re.DOTALL)
    try:
        ans = json_repair.loads(ans)
        assert isinstance(ans, dict), ans
        assert "conditions" in ans and isinstance(ans["conditions"], list), ans
        return ans
    except Exception:
        logging.exception(f"Loading json failure: {ans}")

    return {"conditions": []}


async def gen_json(system_prompt: str, user_prompt: str, chat_mdl, gen_conf={}, max_retry=2, context=""):
    from graphrag.utils import get_llm_cache, set_llm_cache
    cached = get_llm_cache(chat_mdl.llm_name, system_prompt, user_prompt, gen_conf)
    if cached:
        return json_repair.loads(cached)
    _, msg = message_fit_in(form_message(system_prompt, user_prompt), chat_mdl.max_length)
    err = ""
    ans = ""
    prompt_tokens = num_tokens_from_string(system_prompt + user_prompt)
    # Get actual model instance
    actual_mdl = getattr(chat_mdl, 'mdl', chat_mdl)
    base_url = getattr(getattr(actual_mdl, 'async_client', None), '_base_url', 'unknown')
    logging.info(f"[gen_json] Context: {context}, Model: {chat_mdl.llm_name}, Base URL: {base_url}, Prompt tokens: {prompt_tokens}, Max retry: {max_retry}")

    for attempt in range(max_retry):
        if ans and err:
            msg[-1]["content"] += f"\nGenerated JSON is as following:\n{ans}\nBut exception while loading:\n{err}\nPlease reconsider and correct it."
        try:
            logging.info(f"[gen_json] Context '{context}' - Attempt {attempt + 1}/{max_retry} starting...")
            response = await chat_mdl.async_chat(msg[0]["content"], msg[1:], gen_conf=gen_conf)
            # Handle both (ans, tokens) tuple and single string return
            if isinstance(response, tuple) and len(response) >= 1:
                ans = response[0]
                token_used = response[1] if len(response) > 1 else 0
            else:
                ans = response
                token_used = 0
            logging.info(f"[gen_json] Context '{context}' - Attempt {attempt + 1}/{max_retry} got response, len: {len(ans) if ans else 0}, tokens: {token_used}")

            # Check for empty or error response
            if not ans or ans.strip() == "":
                logging.warning(f"[gen_json] Context '{context}' - Empty response from LLM")
                err += "Empty response\n"
                if attempt < max_retry - 1:
                    await asyncio.sleep(2 ** attempt)
                continue

            # Check for error prefix from LLM
            if ans.startswith("**ERROR**"):
                logging.warning(f"[gen_json] Context '{context}' - LLM returned error: {ans}")
                err += f"{ans}\n"
                if attempt < max_retry - 1:
                    await asyncio.sleep(2 ** attempt)
                continue

            ans = re.sub(r"(^.*</think>|```json\n|```\n*$)", "", ans, flags=re.DOTALL)
            res = json_repair.loads(ans)

            # Validate result is not empty
            if res is None or (isinstance(res, (list, dict)) and len(res) == 0):
                logging.warning(f"[gen_json] Context '{context}' - Parsed result is empty: {res}")
                err += f"Empty parsed result: {res}\n"
                if attempt < max_retry - 1:
                    await asyncio.sleep(2 ** attempt)
                continue

            set_llm_cache(chat_mdl.llm_name, system_prompt, ans, user_prompt, gen_conf)
            logging.info(f"[gen_json] Context '{context}' - Success after attempt {attempt + 1}, result type: {type(res)}")
            return res
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            logging.exception(f"[gen_json] Context '{context}' - Attempt {attempt + 1}/{max_retry} FAILED: [{error_type}] {error_msg}")
            err += f"[{error_type}] {error_msg}\n"
            if attempt < max_retry - 1:
                wait_time = 2 ** attempt  # exponential backoff
                logging.info(f"[gen_json] Context '{context}' - Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)

    logging.error(f"[gen_json] Context '{context}' - ALL {max_retry} ATTEMPTS FAILED. Model: {chat_mdl.llm_name}, Base URL: {base_url}")
    logging.error(f"[gen_json] Context '{context}' - Accumulated errors: {err[:500]}")
    return None


TOC_DETECTION = load_prompt("toc_detection")


async def detect_table_of_contents(page_1024: list[str], chat_mdl):
    from graphrag.utils import chat_limiter
    toc_secs = []
    for i, sec in enumerate(page_1024[:22]):
        try:
            async with chat_limiter:
                ans = await gen_json(PROMPT_JINJA_ENV.from_string(TOC_DETECTION).render(page_txt=sec), "Only JSON please.",
                                     chat_mdl, context="detect_table_of_contents")
            if ans and isinstance(ans, dict) and not ans.get("exists"):
                break
            toc_secs.append(sec)
        except Exception as e:
            logging.exception(f"[TOC] detect_table_of_contents failed at section {i}: {e}")
            break
    return toc_secs


TOC_EXTRACTION = load_prompt("toc_extraction")
TOC_EXTRACTION_CONTINUE = load_prompt("toc_extraction_continue")


async def extract_table_of_contents(toc_pages, chat_mdl):
    from graphrag.utils import chat_limiter
    if not toc_pages:
        return []

    try:
        async with chat_limiter:
            result = await gen_json(PROMPT_JINJA_ENV.from_string(TOC_EXTRACTION).render(toc_page="\n".join(toc_pages)),
                                  "Only JSON please.", chat_mdl, context="extract_table_of_contents")
        if result is None:
            logging.error("[TOC] extract_table_of_contents - LLM returned None")
            return []
        return result
    except Exception as e:
        logging.exception(f"[TOC] extract_table_of_contents failed: {e}")
        return []


async def toc_index_extractor(toc: list[dict], content: str, chat_mdl):
    tob_extractor_prompt = """
    You are given a table of contents in a json format and several pages of a document, your job is to add the physical_index to the table of contents in the json format.

    The provided pages contains tags like <physical_index_X> and <physical_index_X> to indicate the physical location of the page X.

    The structure variable is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.

    The response should be in the following JSON format:
    [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>,
            "physical_index": "<physical_index_X>" (keep the format)
        },
        ...
    ]

    Only add the physical_index to the sections that are in the provided pages.
    If the title of the section are not in the provided pages, do not add the physical_index to it.
    Directly return the final JSON structure. Do not output anything else."""

    from graphrag.utils import chat_limiter
    try:
        prompt = tob_extractor_prompt + '\nTable of contents:\n' + json.dumps(toc, ensure_ascii=False,
                                                                              indent=2) + '\nDocument pages:\n' + content
        async with chat_limiter:
            result = await gen_json(prompt, "Only JSON please.", chat_mdl, context="toc_index_extractor")
        if result is None:
            logging.error("[TOC] toc_index_extractor - LLM returned None")
            return []
        return result
    except Exception as e:
        logging.exception(f"[TOC] toc_index_extractor failed: {e}")
        return []


TOC_INDEX = load_prompt("toc_index")


async def table_of_contents_index(toc_arr: list[dict], sections: list[str], chat_mdl):
    from graphrag.utils import chat_limiter
    if not toc_arr or not sections:
        return []

    toc_map = {}
    for i, it in enumerate(toc_arr):
        k1 = (it["structure"] + it["title"]).replace(" ", "")
        k2 = it["title"].strip()
        if k1 not in toc_map:
            toc_map[k1] = []
        if k2 not in toc_map:
            toc_map[k2] = []
        toc_map[k1].append(i)
        toc_map[k2].append(i)

    for it in toc_arr:
        it["indices"] = []
    for i, sec in enumerate(sections):
        sec = sec.strip()
        if sec.replace(" ", "") in toc_map:
            for j in toc_map[sec.replace(" ", "")]:
                toc_arr[j]["indices"].append(i)

    all_pathes = []

    def dfs(start, path):
        nonlocal all_pathes
        if start >= len(toc_arr):
            if path:
                all_pathes.append(path)
            return
        if not toc_arr[start]["indices"]:
            dfs(start + 1, path)
            return
        added = False
        for j in toc_arr[start]["indices"]:
            if path and j < path[-1][0]:
                continue
            _path = deepcopy(path)
            _path.append((j, start))
            added = True
            dfs(start + 1, _path)
        if not added and path:
            all_pathes.append(path)

    dfs(0, [])
    path = max(all_pathes, key=lambda x: len(x))
    for it in toc_arr:
        it["indices"] = []
    for j, i in path:
        toc_arr[i]["indices"] = [j]
    print(json.dumps(toc_arr, ensure_ascii=False, indent=2))

    i = 0
    while i < len(toc_arr):
        it = toc_arr[i]
        if it["indices"]:
            i += 1
            continue

        if i > 0 and toc_arr[i - 1]["indices"]:
            st_i = toc_arr[i - 1]["indices"][-1]
        else:
            st_i = 0
        e = i + 1
        while e < len(toc_arr) and not toc_arr[e]["indices"]:
            e += 1
        if e >= len(toc_arr):
            e = len(sections)
        else:
            e = toc_arr[e]["indices"][0]

        for j in range(st_i, min(e + 1, len(sections))):
            try:
                async with chat_limiter:
                    ans = await gen_json(PROMPT_JINJA_ENV.from_string(TOC_INDEX).render(
                        structure=it["structure"],
                        title=it["title"],
                        text=sections[j]), "Only JSON please.", chat_mdl, context="table_of_contents_index")
                if ans and isinstance(ans, dict) and ans.get("exist") == "yes":
                    it["indices"].append(j)
                    break
            except Exception as e:
                logging.exception(f"[TOC] table_of_contents_index failed at i={i}, j={j}: {e}")
                continue

        i += 1

    return toc_arr


async def check_if_toc_transformation_is_complete(content, toc, chat_mdl):
    from graphrag.utils import chat_limiter
    prompt = """
    You are given a raw table of contents and a  table of contents.
    Your job is to check if the  table of contents is complete.

    Reply format:
    {{
        "thinking": <why do you think the cleaned table of contents is complete or not>
        "completed": "yes" or "no"
    }}
    Directly return the final JSON structure. Do not output anything else."""

    try:
        prompt = prompt + '\n Raw Table of contents:\n' + content + '\n Cleaned Table of contents:\n' + toc
        async with chat_limiter:
            response = await gen_json(prompt, "Only JSON please.", chat_mdl, context="check_if_toc_transformation_is_complete")
        if response and isinstance(response, dict):
            return response.get('completed', 'no')
        return 'no'
    except Exception as e:
        logging.exception(f"[TOC] check_if_toc_transformation_is_complete failed: {e}")
        return 'no'


async def toc_transformer(toc_pages, chat_mdl):
    from graphrag.utils import chat_limiter
    init_prompt = """
    You are given a table of contents, You job is to transform the whole table of content into a JSON format included table_of_contents.

    The `structure` is the numeric system which represents the index of the hierarchy section in the table of contents. For example, the first section has structure index 1, the first subsection has structure index 1.1, the second subsection has structure index 1.2, etc.
    The `title` is a short phrase or a several-words term.

    The response should be in the following JSON format:
    [
        {
            "structure": <structure index, "x.x.x" or None> (string),
            "title": <title of the section>
        },
        ...
    ],
    You should transform the full table of contents in one go.
    Directly return the final JSON structure, do not output anything else. """

    try:
        toc_content = "\n".join(toc_pages)
        prompt = init_prompt + '\n Given table of contents\n:' + toc_content

        def clean_toc(arr):
            for a in arr:
                a["title"] = re.sub(r"[.·….]{2,}", "", a["title"])

        async with chat_limiter:
            last_complete = await gen_json(prompt, "Only JSON please.", chat_mdl, context="toc_transformer")
        if last_complete is None:
            logging.error("[TOC] toc_transformer - Initial LLM call returned None")
            return []

        if_complete = await check_if_toc_transformation_is_complete(toc_content,
                                                                    json.dumps(last_complete, ensure_ascii=False, indent=2),
                                                                    chat_mdl)
        clean_toc(last_complete)
        if if_complete == "yes":
            return last_complete

        max_iterations = 5
        iteration = 0
        while not (if_complete == "yes") and iteration < max_iterations:
            iteration += 1
            prompt = f"""
            Your task is to continue the table of contents json structure, directly output the remaining part of the json structure.
            The response should be in the following JSON format:

            The raw table of contents json structure is:
            {toc_content}

            The incomplete transformed table of contents json structure is:
            {json.dumps(last_complete[-24:], ensure_ascii=False, indent=2)}

            Please continue the json structure, directly output the remaining part of the json structure."""
            async with chat_limiter:
                new_complete = await gen_json(prompt, "Only JSON please.", chat_mdl, context="toc_transformer_continue")
            if not new_complete or str(last_complete).find(str(new_complete)) >= 0:
                break
            clean_toc(new_complete)
            last_complete.extend(new_complete)
            if_complete = await check_if_toc_transformation_is_complete(toc_content,
                                                                        json.dumps(last_complete, ensure_ascii=False,
                                                                                   indent=2), chat_mdl)

        return last_complete
    except Exception as e:
        logging.exception(f"[TOC] toc_transformer failed: {e}")
        return []


TOC_LEVELS = load_prompt("assign_toc_levels")


async def assign_toc_levels(toc_secs, chat_mdl, gen_conf={"temperature": 0.2}):
    from graphrag.utils import chat_limiter
    if not toc_secs:
        return []
    # Get actual model instance
    actual_mdl = getattr(chat_mdl, 'mdl', chat_mdl)
    base_url = getattr(getattr(actual_mdl, 'async_client', None), '_base_url', 'unknown')
    logging.info(f"[TOC] assign_toc_levels - TOC sections: {len(toc_secs)}, "
                 f"Model: {chat_mdl.llm_name}, Base URL: {base_url}")
    try:
        async with chat_limiter:
            result = await gen_json(
                PROMPT_JINJA_ENV.from_string(TOC_LEVELS).render(),
                str(toc_secs),
                chat_mdl,
                gen_conf,
                context="assign_toc_levels"
            )
        logging.info(f"[TOC] assign_toc_levels - gen_json returned type: {type(result)}, value: {result}")
        if result and isinstance(result, list) and len(result) > 0:
            return result
        # If LLM returns empty/invalid, use fallback heuristic
        logging.warning(f"[TOC] assign_toc_levels - LLM returned empty/invalid, using fallback heuristic")
    except Exception as e:
        logging.exception(f"[TOC] assign_toc_levels failed: {e}")

    # Fallback: simple heuristic - first item is level 1, use keywords to detect level changes
    fallback_result = []
    current_level = "1"
    for i, title in enumerate(toc_secs):
        if i == 0:
            current_level = "1"
        # Detect potential chapter/section keywords
        elif any(kw in title.lower() for kw in ['chapter', 'part', '附录', 'chapter', '部分']):
            current_level = "1"
        elif any(kw in title.lower() for kw in ['section', '节', '章', '篇']):
            current_level = "2"
        fallback_result.append({"level": current_level, "title": title})
    #logging.info(f"[TOC] assign_toc_levels - Fallback result: {fallback_result}")
    return fallback_result


TOC_FROM_TEXT_SYSTEM = load_prompt("toc_from_text_system")
TOC_FROM_TEXT_USER = load_prompt("toc_from_text_user")


# Generate TOC from text chunks with text llms
async def gen_toc_from_text(txt_info: dict, chat_mdl, callback=None):
    from graphrag.utils import chat_limiter
    if callback:
        callback(msg="")
    try:
        # Get actual model instance
        actual_mdl = getattr(chat_mdl, 'mdl', chat_mdl)
        base_url = getattr(getattr(actual_mdl, 'async_client', None), '_base_url', 'unknown')
        #logging.info(f"[TOC] gen_toc_from_text - Model: {chat_mdl.llm_name}, Base URL: {base_url}")
        #logging.info(f"[TOC] gen_toc_from_text - Chunks count: {len(txt_info.get('chunks', []))}")
        async with chat_limiter:
            ans = await gen_json(
                PROMPT_JINJA_ENV.from_string(TOC_FROM_TEXT_SYSTEM).render(),
                PROMPT_JINJA_ENV.from_string(TOC_FROM_TEXT_USER).render(
                    text="\n".join([json.dumps(d, ensure_ascii=False) for d in txt_info["chunks"]])),
                chat_mdl,
                gen_conf={"temperature": 0.0, "top_p": 0.9},
                context="gen_toc_from_text"
            )
        if ans is None:
            logging.error(f"[TOC] gen_toc_from_text - LLM returned None, Base URL: {base_url}")
            txt_info["toc"] = []
        else:
            txt_info["toc"] = ans if ans and not isinstance(ans, str) else []
    except Exception as e:
        logging.exception(f"[TOC] gen_toc_from_text failed: {e}")
        txt_info["toc"] = []


def split_chunks(chunks, max_length: int):
    """
    Pack chunks into batches according to max_length, returning [{"id": idx, "text": chunk_text}, ...].
    Do not split a single chunk, even if it exceeds max_length.
    """

    result = []
    batch, batch_tokens = [], 0

    for idx, chunk in enumerate(chunks):
        t = num_tokens_from_string(chunk)
        if batch_tokens + t > max_length:
            result.append(batch)
            batch, batch_tokens = [], 0
        batch.append({idx: chunk})
        batch_tokens += t
    if batch:
        result.append(batch)
    return result


async def run_toc_from_text(chunks, chat_mdl, callback=None):
    #from graphrag.utils import chat_limiter
    input_budget = int(chat_mdl.max_length * INPUT_UTILIZATION) - num_tokens_from_string(
        TOC_FROM_TEXT_USER + TOC_FROM_TEXT_SYSTEM
    )

    input_budget = 1024 if input_budget > 1024 else input_budget
    chunk_sections = split_chunks(chunks, input_budget)
    titles = []

    # Get actual model instance
    #actual_mdl = getattr(chat_mdl, 'mdl', chat_mdl)
    #base_url = getattr(getattr(actual_mdl, 'async_client', None), '_base_url', 'unknown')
    #logging.info(f"[TOC] run_toc_from_text - Total chunks: {len(chunks)}, Split into: {len(chunk_sections)} sections, "f"Model: {chat_mdl.llm_name}, Base URL: {base_url}")

    chunks_res = []
    tasks = []
    for i, chunk in enumerate(chunk_sections):
        if not chunk:
            continue
        chunks_res.append({"chunks": chunk})
        #logging.info(f"[TOC] Creating task {i+1}/{len(chunk_sections)} with {len(chunk)} chunks")
        tasks.append(asyncio.create_task(gen_toc_from_text(chunks_res[-1], chat_mdl, callback)))
    try:
        await asyncio.gather(*tasks, return_exceptions=False)
        logging.info(f"[TOC] All {len(tasks)} tasks completed successfully")
    except Exception as e:
        logging.error(f"[TOC] Error generating TOC: {e}")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    for chunk in chunks_res:
        titles.extend(chunk.get("toc", []))

    # Filter out entries with title == -1
    prune = len(titles) > 512
    max_len = 12 if prune else 22
    filtered = []
    for x in titles:
        if not isinstance(x, dict) or not x.get("title") or x["title"] == "-1":
            continue
        if len(rag_tokenizer.tokenize(x["title"]).split(" ")) > max_len:
            continue
        if re.match(r"[0-9,.()/ -]+$", x["title"]):
            continue
        filtered.append(x)

    #logging.info(f"\n\nFiltered TOC sections:\n{filtered}")
    if not filtered:
        return []

    # Generate initial level (level/title)
    raw_structure = [x.get("title", "") for x in filtered]
    #logging.info(f"[TOC] run_toc_from_text - Raw structure for level assignment: {raw_structure}")

    # Assign hierarchy levels using LLM
    toc_with_levels = await assign_toc_levels(raw_structure, chat_mdl, {"temperature": 0.0, "top_p": 0.9})
    #logging.info(f"[TOC] run_toc_from_text - assign_toc_levels returned: {toc_with_levels}")
    if not toc_with_levels:
        logging.error("[TOC] run_toc_from_text - assign_toc_levels returned empty, returning []")
        return []

    # Merge structure and content (by index)
    logging.info(f"[TOC] run_toc_from_text - Merging: toc_with_levels({len(toc_with_levels)}) with filtered({len(filtered)})")
    if len(toc_with_levels) != len(filtered):
        logging.error(f"[TOC] run_toc_from_text - MISMATCH: toc_with_levels has {len(toc_with_levels)} items, filtered has {len(filtered)} items")
        # Try to match by title
        toc_map = {item.get('title', ''): item for item in toc_with_levels if isinstance(item, dict)}
        toc_with_levels = [toc_map.get(src.get('title', ''), {'level': '1', 'title': src.get('title', '')}) for src in filtered]

    prune = len(toc_with_levels) > 512
    max_lvl = "0"
    sorted_list = sorted([t.get("level", "0") for t in toc_with_levels if isinstance(t, dict)])
    if sorted_list:
        max_lvl = sorted_list[-1]
    merged = []
    for i, (toc_item, src_item) in enumerate(zip(toc_with_levels, filtered)):
        if not isinstance(toc_item, dict):
            logging.warning(f"[TOC] run_toc_from_text - Item {i} is not a dict: {toc_item}")
            continue
        if prune and toc_item.get("level", "0") >= max_lvl:
            continue
        merged.append({
            "level": toc_item.get("level", "0"),
            "title": toc_item.get("title", ""),
            "chunk_id": src_item.get("chunk_id", ""),
        })

    logging.info(f"[TOC] run_toc_from_text - Merged result: {len(merged)} items")
    return merged


TOC_RELEVANCE_SYSTEM = load_prompt("toc_relevance_system")
TOC_RELEVANCE_USER = load_prompt("toc_relevance_user")
async def relevant_chunks_with_toc(query: str, toc: list[dict], chat_mdl, topn: int = 6):
    from graphrag.utils import chat_limiter
    import numpy as np
    
    if not toc:
        return []
    
    # Limit TOC size to avoid large request body and timeout
    # Model server takes ~100s to process large TOC, causing connection drops
    MAX_TOC_ITEMS = 30  # Reduced to ensure processing completes within connection timeout
    MAX_TOC_LEVEL = 2   # Only keep top-level sections (level 1-2) for faster processing
    original_count = len(toc)
    
    # Filter: keep only level 1-2 entries, they are most relevant for retrieval
    toc = [t for t in toc if int(t.get("level", 1)) <= MAX_TOC_LEVEL]
    level_filtered_count = len(toc)
    
    if len(toc) > MAX_TOC_ITEMS:
        logging.warning(f"[TOC] relevant_chunks_with_toc - TOC too large ({original_count}, level 1-2: {level_filtered_count}), truncating to {MAX_TOC_ITEMS}")
        toc = toc[:MAX_TOC_ITEMS]
    else:
        logging.info(f"[TOC] relevant_chunks_with_toc - Filtered to level 1-2: {level_filtered_count}/{original_count} entries")
    
    # Estimate prompt size
    toc_json_str = "[\n%s\n]" % "\n".join(
        [json.dumps({"level": d["level"], "title": d["title"]}, ensure_ascii=False) for d in toc]
    )
    prompt_size = len(toc_json_str) + len(query)
    logging.info(f"[TOC] relevant_chunks_with_toc - Query: '{query[:50]}...', TOC items: {len(toc)}/{original_count}, "
                 f"Prompt size: {prompt_size} chars, Model: {chat_mdl.llm_name if chat_mdl else 'None'}")
    
    try:
        async with chat_limiter:
            ans = await gen_json(
                PROMPT_JINJA_ENV.from_string(TOC_RELEVANCE_SYSTEM).render(),
                PROMPT_JINJA_ENV.from_string(TOC_RELEVANCE_USER).render(query=query, toc_json=toc_json_str + "\n"),
                chat_mdl,
                gen_conf={"temperature": 0.0, "top_p": 0.9},
                context="relevant_chunks_with_toc"
            )
        id2score = {}
        for ti, sc in zip(toc, ans):
            if not isinstance(sc, dict) or sc.get("score", -1) < 1:
                continue
            for id in ti.get("ids", []):
                if id not in id2score:
                    id2score[id] = []
                id2score[id].append(sc["score"] / 5.)
        for id in id2score.keys():
            id2score[id] = np.mean(id2score[id])
        return [(id, sc) for id, sc in list(id2score.items()) if sc >= 0.3][:topn]
    except Exception as e:
        logging.exception(e)
    return []


META_DATA = load_prompt("meta_data")
async def gen_metadata(chat_mdl, schema: dict, content: str):
    template = PROMPT_JINJA_ENV.from_string(META_DATA)
    for k, desc in schema["properties"].items():
        if "enum" in desc and not desc.get("enum"):
            del desc["enum"]
        if desc.get("enum"):
            desc["description"] += "\n** Extracted values must strictly match the given list specified by `enum`. **"
    system_prompt = template.render(content=content, schema=schema)
    user_prompt = "Output: "
    _, msg = message_fit_in(form_message(system_prompt, user_prompt), chat_mdl.max_length)
    ans = await chat_mdl.async_chat(msg[0]["content"], msg[1:])
    return re.sub(r"^.*</think>", "", ans, flags=re.DOTALL)


SUFFICIENCY_CHECK = load_prompt("sufficiency_check")
async def sufficiency_check(chat_mdl, question: str, ret_content: str):
    try:
        return await gen_json(
            PROMPT_JINJA_ENV.from_string(SUFFICIENCY_CHECK).render(question=question, retrieved_docs=ret_content),
            "Output:\n",
            chat_mdl
        )
    except Exception as e:
        logging.exception(e)
    return {}


MULTI_QUERIES_GEN = load_prompt("multi_queries_gen")
async def multi_queries_gen(chat_mdl, question: str, query:str, missing_infos:list[str], ret_content: str):
    try:
        return await gen_json(
            PROMPT_JINJA_ENV.from_string(MULTI_QUERIES_GEN).render(
                original_question=question,
                original_query=query,
                missing_info="\n - ".join(missing_infos),
                retrieved_docs=ret_content
            ),
            "Output:\n",
            chat_mdl
        )
    except Exception as e:
        logging.exception(e)
    return {}