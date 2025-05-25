import os
import re
import glob
import json
import pickle
import argparse
from time import time
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Set

RED, YELLOW, GREEN, RESET = "\033[31m", "\033[33m", "\033[32m", "\033[0m"


class FilenameKeywordPromptBuilder:

    def __init__(self):
        self.system_prompt = (
            "You are a creative but grounded assistant. Given a keyword that implies a certain sound, primarily animal sounds or occasionally musical instruments, "
            "you must generate at least 20 distinct English expressions that vividly evoke or describe that sound, separated by '|'. "
            "Do not number the expressions. Separate each expression with a vertical bar (|)."
            "The first 10 expressions must be concise, using 2 to 4 words each, and stay closely related to the keyword. "
            "The next 10 expressions must be gradually longer, ranging from 4 to 10 words, and remain semantically close to the keyword. "
            "For animal sounds, focus on the specific vocalization or action (e.g., meowing for cats, chirping for birds). "
            "For musical instruments, emphasize the characteristic sound, such as plucking for string instruments like sitar. "
            "Use rich, varied language to avoid repetition, incorporating specific contexts (e.g., natural habitats for animals, musical settings for instruments). "
            "Ensure all expressions are unique, realistic, and evocative of the sound’s perception. "
            "Do not generate fewer than 20 expressions, and strictly adhere to the format. "
            "If possible, extend the list up to a maximum of 50 expressions following the same pattern.")

        self.initial_example = {
            "role":
                "user",
            "content":
                "Describe the sound implied by the keyword: Animal_Wild animals_Bird_Bird vocalization_bird call_bird song_Chirp_tweet"
        }

        self.initial_response = {
            "role":
                "assistant",
            "content": (
                "bird chirping|gentle tweet|morning chirps|Chirp chirp|Tweet tweet|Sharp trill|Quick cheep|High-pitched warble|"
                "Soft whistle|Bright twitter|"
                "soft chirping sounds coming from the leafy trees|melodious bird calls at the break of dawn|short high-pitched tweets fading into silence|"
                "layered chirping echoing softly in the distance|birds singing joyfully on sunlit tree branches|sweet bird songs resonating through the quiet forest|"
                "Sparrow’s cheerful chirp echoing warmly in the garden|Finch tweets gently at dawn’s early light|Warbler’s trill piercing through the morning mist|"
                "Robin’s sharp cheep greeting the fresh sunrise|a series of rapid, gentle chirps in cool morning air|distinct warbling notes in early spring mornings|"
                "soft twittering sounds amidst the rustling leaves|harmonious bird songs blending with the gentle breeze|high-pitched chirps slowly fading at dusk|"
                "playful bird calls near the shimmering lake surface|clear melodious whistles echoing in the quiet garden|gentle tweets floating through the open valley|"
                "layered bird songs creating a natural morning symphony|bright, sharp chirps ringing through the blooming meadow|melodic calls carried by the gentle wind|"
                "birdsong echoing softly through the forest canopy|short bursts of cheerful tweets in sunny clearings|sweet warbles from hidden perches in the trees|"
                "rhythmic chirping beneath the warm morning sun|soft whistles accompanied by the rustling of leaves|delicate bird calls drifting across the open field|"
                "gentle trills weaving softly through leafy branches|clear, sharp notes from small perched songbirds|gentle morning chirps filling the fresh air|"
                "quiet twittering at the break of early dawn|light, playful chirps fluttering among colorful flowers")
        }
        self.instrument_example = {
            "role": "user",
            "content": "Describe the sound implied by the keyword: Music_Musical instrument_Plucked string instrument_Sitar"
        }
        self.instrument_response = {
            "role":
                "assistant",
            "content": (
                "plucked sitar|sharp twang|gentle pluck|sitar strum|vibrant twang|quick plink|resonant pluck|sitar’s chime|soft plink|bright twang|"
                "plucked strings resonate softly|sitar’s sharp twang echoes|gentle plucks fill silence|vibrant sitar notes dance|sitar strings hum melodically|"
                "quick plinks in quiet room|sitar’s resonant plucks soar|twanging sitar in evening air|plucked sitar strings vibrate gently|sitar’s bright twang at dusk"
            )
        }
        self.domestic_example = {
            "role": "user",
            "content": "Describe the sound implied by the keyword: Sounds of things_Domestic sounds_home sounds_Toilet flush."
        }
        self.domestic_response = {
            "role":
                "assistant",
            "content": (
                "toilet flush|water whoosh|gurgling drain|flush valve|swirling water|rapid gush|draining slosh|flush surge|water swirl|suction gurgle|"
                "water rushes in toilet bowl|gurgles echo in bathroom drain|swirling flush fills quiet room|rapid whoosh of toilet tank|valve releases surging water|"
                "slosh and gurgle in pipes|toilet’s flush echoes in house|water swirls down ceramic bowl|drain’s suction pulls water away|flush’s loud surge at night"
            )
        }

    def get_keyword_from_filename(self, filename: str) -> str:
        name_without_ext = filename.rsplit('.', 1)[0]
        match = re.match(r'^\d+_\d+(.*)', name_without_ext)
        if match:
            return match.group(1)
        else:
            return name_without_ext

    def keyword_to_messages(self, keyword: str) -> List[Dict[str, str]]:
        return [{
            "role": "system",
            "content": self.system_prompt
        }, self.initial_example, self.initial_response, self.instrument_example, self.instrument_response,
                self.domestic_example, self.domestic_response, {
                    "role": "user",
                    "content": f"Describe the sound implied by the keyword: {keyword}."
                }, {
                    "role": "assistant",
                    "content": ""
                }]

    def generate_messages(self, filename_basename: str) -> List[Dict[str, str]]:
        keyword = self.get_keyword_from_filename(filename_basename)
        return self.keyword_to_messages(keyword)


PromptBuilder = FilenameKeywordPromptBuilder()


class FileNameDataset(Dataset):

    def __init__(self, basenames: List[str]):
        self.basenames = basenames

    def __len__(self):
        return len(self.basenames)

    def __getitem__(self, index):
        filename_basename = self.basenames[index]
        return filename_basename, PromptBuilder.generate_messages(filename_basename)


# --- New helper function for cleaning LLM descriptions ---
def _clean_llm_description(raw_text: str) -> List[str]:
    """
    Cleans a raw LLM output string into a list of individual, unique descriptions.
    Handles cases where descriptions are separated by '|' OR by leading numbers.
    """
    # Remove assistant prefix and normalize general whitespace
    cleaned_text = re.sub(r"^\[assistant\]\s*", "", raw_text)
    cleaned_text = cleaned_text.replace('\n', ' ').replace('\r', ' ')
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    individual_descriptions = [desc.strip() for desc in cleaned_text.split('|') if desc.strip()]
    if len(individual_descriptions) <= 1 and re.search(r"\d+\.\s", cleaned_text):
        # Split by numbered list pattern
        pattern_numbered_list = r"\d+\.\s*(.*?)(?=\d+\.|$)"
        extracted_descriptions = re.findall(pattern_numbered_list, cleaned_text)

        individual_descriptions = [desc.strip() for desc in extracted_descriptions if desc.strip()]
        if not individual_descriptions:
            # Fallback if neither split worked as expected, just return the whole cleaned text as one item.
            individual_descriptions = [cleaned_text] if cleaned_text else []

    # Remove leading numbers (e.g., "1. ", "25 ") from each description (applies to both split methods)
    final_cleaned_descriptions = []
    for desc in individual_descriptions:
        # This line was already robust and will now handle both cases (pre-existing numbers in | separated, or numbers from list split)
        cleaned_desc = re.sub(r"^\d+\.?\s*", "", desc).strip()
        final_cleaned_descriptions.append(cleaned_desc)

    return final_cleaned_descriptions


# --- End of new helper function ---


def main(config):
    os.environ["HF_HOME"] = config.llm_dir
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=40,
        max_tokens=config.max_new_tokens,
        stop=["Describe the sound implied by the keyword:", "[user]", "[system]"],
        n=1,
    )

    llm = LLM(model=config.llm_model,
              dtype="auto",
              quantization="awq" if config.low_bit else None,
              tensor_parallel_size=config.num_gpus,
              enable_prefix_caching=not config.disable_apc)

    root_dir = os.path.join(config.dataset_path, 'Vim_Sketch_Dataset')
    all_full_audio_paths = [
        os.path.abspath(f)
        for f in glob.glob(os.path.join(root_dir, '**', '*'), recursive=True)
        if os.path.splitext(f)[1].lower() in {'.wav', '.mp3', '.flac'}
    ]
    audio_basenames = sorted(list(set(os.path.basename(f) for f in all_full_audio_paths)))

    tqdm.write(f"Found {len(audio_basenames)} unique audio basenames in {root_dir}.")
    tqdm.write(
        f"{RED}NOTE{RESET}: This generates sound descriptions using both reference and imitation file names in the {RED}style of reference{RESET} descriptions."
    )

    logger = {}
    final_descriptions = {}
    descriptions_in_progress = {b: set() for b in audio_basenames}
    files_to_process = set(audio_basenames)
    attempt = 0

    tqdm.write(f"Targeting {config.target_desc_count} unique descriptions per file.")
    start = time()

    while files_to_process and attempt < config.max_tries:
        attempt += 1
        tqdm.write(f"\n--- Attempt {attempt} of {config.max_tries} ---")

        current_run_basenames = list(files_to_process)
        if not current_run_basenames:
            tqdm.write("No files left to process in this attempt.")
            break

        ds = FileNameDataset(current_run_basenames)
        dl = DataLoader(ds,
                        batch_size=config.batch_size,
                        shuffle=False,
                        num_workers=0,
                        drop_last=False,
                        collate_fn=lambda batch: ([item[0] for item in batch], [item[1] for item in batch]))

        tqdm.write(f"Processing {len(current_run_basenames)} files in this attempt...")

        sample_output_for_batch = None
        sample_filename_for_batch = None

        for batch_idx, (filenames_in_batch, messages_batch) in enumerate(tqdm(dl, desc=f"Attempt {attempt}")):
            # # For testing purposes, break after the first batch
            # if batch_idx > 0:
            #     break

            def serialize_chat(messages):
                return "\n".join(
                    f"[{m.get('role', 'unknown')}] {m.get('content', '')}" for m in messages if isinstance(m, dict))

            prompts = [serialize_chat(msg) for msg in messages_batch]
            outputs = llm.generate(prompts, sampling_params)

            for filename_basename, output_obj in zip(filenames_in_batch, outputs):
                if filename_basename not in files_to_process:
                    continue

                if not (hasattr(output_obj, 'outputs') and output_obj.outputs and hasattr(output_obj.outputs[0], 'text')):
                    tqdm.write(f"Warning: Malformed output for {filename_basename} during processing. Skipping.")
                    continue

                result_text = output_obj.outputs[0].text.strip()

                # --- Use the new helper function for cleaning ---
                generated_descriptions_in_batch = _clean_llm_description(result_text)

                # Capture sample output if not already captured for this batch
                # The sample will now also be fully cleaned, including number removal
                if sample_output_for_batch is None:
                    sample_output_for_batch = '|'.join(generated_descriptions_in_batch)
                    sample_filename_for_batch = filename_basename

                if not generated_descriptions_in_batch:
                    continue

                newly_added_count = 0
                for desc in generated_descriptions_in_batch:
                    if len(descriptions_in_progress[filename_basename]) < config.target_desc_count:
                        if desc not in descriptions_in_progress[filename_basename]:
                            descriptions_in_progress[filename_basename].add(desc)
                            newly_added_count += 1
                    else:
                        break

                if newly_added_count > 0:
                    tqdm.write(
                        f"File: {filename_basename} - Added {newly_added_count} new unique. Total unique: {len(descriptions_in_progress[filename_basename])}/{config.target_desc_count}"
                    )

                if len(descriptions_in_progress[filename_basename]) >= config.target_desc_count:
                    final_descriptions[filename_basename] = list(
                        descriptions_in_progress[filename_basename])[:config.target_desc_count]
                    files_to_process.discard(filename_basename)
                    tqdm.write(
                        f"{GREEN}COMPLETED:{RESET} {filename_basename}. Collected {len(final_descriptions[filename_basename])} unique descriptions."
                    )
            if sample_output_for_batch is not None:
                tqdm.write('\n--- Sample Output for the Processed Batch ---')
                tqdm.write(f"Sample File: {RED}{sample_filename_for_batch}{RESET}")
                tqdm.write(f"  Cleaned Text Sample: \"{YELLOW}{sample_output_for_batch[:250]}...\"{RESET}")
                tqdm.write('-------------------------------------------\n')

        logger[attempt] = {
            "remaining_files": len(files_to_process),
            "collected_files": len(audio_basenames) - len(files_to_process),
            "collected_descriptions": sum(len(v) for v in descriptions_in_progress.values()),
            "total_descriptions": sum(len(v) for v in final_descriptions.values())
        }
        # # Check if we need to break out of the loop, just for test purposes
        #     break
        # break

    tqdm.write(f"--- End of Attempt {attempt} ---")
    logger[attempt] = {
        "remaining_files": len(files_to_process),
        "collected_files": len(audio_basenames) - len(files_to_process),
        "collected_descriptions": sum(len(v) for v in descriptions_in_progress.values()),
        "total_descriptions": sum(len(v) for v in final_descriptions.values())
    }
    if files_to_process:
        tqdm.write(f"{len(files_to_process)} files still require descriptions.")
    else:
        tqdm.write("All files have reached the target number of descriptions.")

    if files_to_process:
        tqdm.write(
            f"\n🛑 Failed to collect {config.target_desc_count} unique descriptions for {len(files_to_process)} files after {attempt} attempts:"
        )
        for f_name in files_to_process:
            collected_count = len(descriptions_in_progress[f_name])
            tqdm.write(f"- {f_name} (Collected {collected_count}/{config.target_desc_count} unique descriptions)")
            if collected_count > 0 and f_name not in final_descriptions:
                final_descriptions[f_name] = list(descriptions_in_progress[f_name])
                tqdm.write(f"  Stored {collected_count} descriptions for {f_name}.")

    else:
        tqdm.write(f"\n✅ Successfully collected {config.target_desc_count} unique descriptions for all processed files.")

    elapsed = time() - start
    config.elapse_time = f"Time elapsed: {int(elapsed // 60)}m {int(elapsed % 60)}s"
    tqdm.write(config.elapse_time)

    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    with open(config.output_path, "w", encoding="utf-8") as f:
        json.dump(final_descriptions, f, ensure_ascii=False, indent=1)
    with open(config.output_path_pkl, "wb") as f:
        pickle.dump(final_descriptions, f)
    with open(config.output_path.replace('.json', '.log'), "w") as f:
        json.dump(vars(config), f, indent=2)
        f.write("\n\n")
        json.dump(logger, f, indent=2)
    tqdm.write(f"Saved the generated descriptions to {RED}{config.output_path}{RESET}.")
    tqdm.write(f"Saved logs to {RED}{config.output_path.replace('.json', '.log')}{RESET}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument parser for reference-style LLM-prompt generation on the VimSketch.")

    parser.add_argument('--dataset_path', type=str, default='data', help="Path to the data sets.")
    parser.add_argument('--output_path',
                        type=str,
                        default='',
                        help="Default: data/Vim_Sketch_Dataset/s2s/descriptions/references.json")

    parser.add_argument('--target_desc_count', type=int, default=50, help="Target number of unique descriptions per file.")
    parser.add_argument('--max_tries', type=int, default=10, help="Maximum number of attempts to target description count.")

    parser.add_argument('--llm_dir', type=str, default='src/prepare_s2s_data/LLM', help="Path to the vLLM cache directory.")
    parser.add_argument('--llm_model', type=str, default='microsoft/Phi-4', help="LLM model name.")
    parser.add_argument('--num_gpus', type=int, default=1, help="Number of GPUs. Default: 1")
    parser.add_argument('--batch_size', type=int, default=2048, help="Set high number, then vLLM dynamically adjusts.")
    parser.add_argument('--max_new_tokens', type=int, default=1200, help="Maximum number of new tokens to generate.")
    parser.add_argument('--return_full_text', type=bool, default=False, help="Whether to return including prompt.")
    parser.add_argument('--temperature', type=float, default=0.8, help="Sampling temperature for generation.")
    parser.add_argument('--top_p', type=float, default=0.8, help="Top-p sampling for generation.")
    parser.add_argument('--do_sample', default=True, action='store_true', help="Whether to use sampling.")
    parser.add_argument('--low_bit', default=False, action='store_true', help="Whether to use 4-bit quant.")
    parser.add_argument('--disable_apc', default=False, action='store_true', help="Disable Automatic Prefix Caching).")
    args = parser.parse_args()

    if not args.output_path:
        model_short_name = args.llm_model.split('/')[-1]
        temperature = f"{int(args.temperature * 100):03d}"
        timestamp = datetime.now().strftime("%y%m%d%H%M")
        args.output_path = f'data/Vim_Sketch_Dataset/s2s/descriptions/references_{model_short_name}_t{temperature}_{timestamp}.json'
        args.output_path_pkl = f'data/Vim_Sketch_Dataset/s2s/descriptions/references_{model_short_name}_t{temperature}_{timestamp}.pkl'

    main(args)
