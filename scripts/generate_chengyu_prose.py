#!/usr/bin/env python3
"""Generate coherent modern Chinese prose for the chengyu corpus.

Reads bare chengyu from chinese_classics_combined.txt, sends batches to
Claude to weave into flowing essays that illustrate each idiom's meaning,
then rewrites the corpus with the generated prose.

Results are cached to disk after each batch so failures don't waste API work.
"""

import anthropic
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BATCH_SIZE = 20
MAX_WORKERS = 5
CORPUS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "corpora", "chinese_classics"
)
CACHE_FILE = os.path.join(CORPUS_DIR, ".chengyu_cache.json")

PROMPT_TEMPLATE = """请用以下成语写一段连贯的现代汉语散文。要求：
1. 你可以自由调整成语顺序，使文章更加流畅连贯
2. 每个成语必须在文中出现，且用法正确、自然，能体现其含义
3. 文章应像一篇有主题的散文或随笔，读起来浑然一体，不是造句练习
4. 用【】标出每个成语以便核对
5. 只输出文章正文，不要加标题或其他说明

成语（{count}个）：{chengyu_list}"""


def read_chengyu(path):
    with open(path) as f:
        lines = f.readlines()
    chengyu = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if len(s) <= 4:
            chengyu.append(s)
        else:
            break
    return chengyu


def read_prose_offset(path):
    """Return the line index where classical prose begins (after chengyu section)."""
    with open(path) as f:
        for i, line in enumerate(f):
            s = line.strip()
            if s and len(s) > 4:
                return i
    return 0


def strip_markers(text):
    """Remove 【】 markers from generated text."""
    return text.replace("【", "").replace("】", "")


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def generate_batch(client, chengyu_batch, batch_idx):
    prompt = PROMPT_TEMPLATE.format(
        count=len(chengyu_batch),
        chengyu_list="、".join(chengyu_batch),
    )
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            missing = [c for c in chengyu_batch if c not in text]
            if missing:
                print(
                    f"  Batch {batch_idx+1}: missing {len(missing)} chengyu "
                    f"({', '.join(missing[:3])}...), retry {attempt+1}",
                    flush=True,
                )
                continue
            return (batch_idx, strip_markers(text))
        except Exception as e:
            print(f"  Batch {batch_idx+1} attempt {attempt+1}: {e}", flush=True)
            time.sleep(2 ** attempt)
    return (batch_idx, None)


def recover_batch_individually(client, chengyu_batch):
    """Fall back to generating one sentence per chengyu."""
    sentences = []
    for c in chengyu_batch:
        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=500,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"请用成语「{c}」造一个自然流畅的现代汉语句子，"
                            f"体现该成语的含义。只输出句子本身，不要加引号或编号。"
                        ),
                    }],
                )
                sentence = strip_markers(resp.content[0].text.strip())
                if c in sentence:
                    sentences.append(sentence)
                    break
            except Exception as e:
                time.sleep(2 ** attempt)
    return " ".join(sentences) if len(sentences) == len(chengyu_batch) else None


def main():
    combined_path = os.path.join(CORPUS_DIR, "chinese_classics_combined.txt")

    chengyu = read_chengyu(combined_path)
    print(f"Found {len(chengyu)} chengyu")

    batches = [chengyu[i : i + BATCH_SIZE] for i in range(0, len(chengyu), BATCH_SIZE)]
    print(f"Split into {len(batches)} batches of ~{BATCH_SIZE}")

    # Load cached results from previous runs
    cache = load_cache()
    client = anthropic.Anthropic()

    # Determine which batches still need generation
    todo = []
    for i, batch in enumerate(batches):
        key = str(i)
        if key in cache and cache[key] is not None:
            continue
        todo.append((i, batch))

    if todo:
        print(f"{len(todo)} batches to generate ({len(batches) - len(todo)} cached)")
        done_count = len(batches) - len(todo)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(generate_batch, client, batch, i): i
                for i, batch in todo
            }
            for future in as_completed(futures):
                idx, text = future.result()
                done_count += 1
                if text:
                    cache[str(idx)] = text
                    save_cache(cache)
                    print(f"[{done_count}/{len(batches)}] Batch {idx+1} ok", flush=True)
                else:
                    print(f"[{done_count}/{len(batches)}] Batch {idx+1} FAILED", flush=True)
    else:
        print("All batches cached.")

    # Recover failed batches one chengyu at a time
    failed_indices = [i for i in range(len(batches)) if str(i) not in cache or cache[str(i)] is None]
    if failed_indices:
        print(f"\n{len(failed_indices)} batches failed. Recovering individually...", flush=True)
        for idx in failed_indices:
            print(f"  Recovering batch {idx+1} ({len(batches[idx])} chengyu)...", flush=True)
            text = recover_batch_individually(client, batches[idx])
            if text:
                cache[str(idx)] = text
                save_cache(cache)
                print(f"  Recovered batch {idx+1}", flush=True)
            else:
                print(f"  Batch {idx+1} could not be recovered", flush=True)

    # Final check
    results = []
    for i in range(len(batches)):
        key = str(i)
        if key in cache and cache[key] is not None:
            results.append(cache[key])
        else:
            print(f"FATAL: Batch {i+1} has no result. Aborting.")
            sys.exit(1)

    # Read the classical prose section (everything after chengyu)
    prose_offset = read_prose_offset(combined_path)
    with open(combined_path) as f:
        all_lines = f.readlines()
    classical_prose = all_lines[prose_offset:]

    # Rewrite: generated chengyu prose + original classical prose
    with open(combined_path, "w") as f:
        for r in results:
            f.write(r.strip() + "\n")
        f.write("\n")
        f.writelines(classical_prose)

    total_chengyu_chars = sum(len(r) for r in results)
    print(f"\nDone. {len(results)} essays, ~{total_chengyu_chars:,} chars of chengyu prose.")
    print(f"Classical prose section preserved ({len(classical_prose)} lines).")

    # Clean up cache
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)

    # Update sources.tsv
    update_sources(results)


def update_sources(results):
    """Rewrite the chengyu-catalog entries in sources.tsv."""
    tsv_path = os.path.join(CORPUS_DIR, "sources.tsv")
    with open(tsv_path) as f:
        lines = f.readlines()

    header = lines[0]
    other_lines = [l for l in lines[1:] if not l.startswith("chengyu-catalog")]

    new_entries = []
    for i, essay in enumerate(results):
        char_count = len(essay.strip())
        new_entries.append(
            f"chengyu-catalog\tchapter-{i+1:03d}\tessay-{i+1:03d}\t{char_count}\n"
        )

    with open(tsv_path, "w") as f:
        f.write(header)
        for entry in new_entries:
            f.write(entry)
        for line in other_lines:
            f.write(line)

    print(f"Updated sources.tsv: {len(new_entries)} chengyu essays + {len(other_lines)} classical entries.")


if __name__ == "__main__":
    main()
