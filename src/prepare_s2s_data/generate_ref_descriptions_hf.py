from typing import List, Dict
import os
import re
import glob
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# debug
# from types import SimpleNamespace
# config = SimpleNamespace(dataset_path="data", llm_dir='src/prepare_s2s_data/LLM', llm_model='microsoft/Phi-4-mini-instruct', batch_size=2, num_workers=2)
RED, YELLOW, GREEN, RESET = "\033[31m", "\033[33m", "\033[32m", "\033[0m"


class FilenameKeywordPromptBuilder:

    def __init__(self):
        self.system_prompt = (
            "You are a creative but grounded assistant. Given a keyword that implies a certain sound, primarily animal sounds or occasionally musical instruments, "
            "you must generate at least 20 distinct English expressions that vividly evoke or describe that sound, separated by '|'. "
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
        }, self.initial_example, self.initial_response, {
            "role": "user",
            "content": f"Describe the sound implied by the keyword: {keyword}"
        }]

    def generate_messages(self, filename: str) -> List[Dict[str, str]]:
        keyword = self.get_keyword_from_filename(filename)
        return self.keyword_to_messages(keyword)


def main(config):
    # Set the vLLM cache directory
    os.environ["HF_HOME"] = config.llm_dir

    # LLM: why import here? <-- need to import after setting a custom HF_HOME directory
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    model = AutoModelForCausalLM.from_pretrained(
        config.llm_model,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
        load_in_4bit=config.low_bit,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model, padding_side="left")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": config.max_new_tokens,
        "return_full_text": config.return_full_text,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "do_sample": config.do_sample,
    }

    # # Class object that converts filenames to LLM prompts to generate sound descriptions
    # PromptBuilder = FilenameKeywordPromptBuilder()

    # Collect all audiofile paths: VimSketch
    root_dir = os.path.join(config.dataset_path, 'Vim_Sketch_Dataset')
    audiofiles = [
        os.path.abspath(f)
        for f in glob.glob(os.path.join(root_dir, '**', '*'), recursive=True)
        if os.path.splitext(f)[1].lower() in {'.wav', '.mp3', '.flac'}
    ]
    print(f"Found {len(audiofiles)} audio files in {root_dir}.")
    print(
        f"{RED}NOTE{RESET}: This generates sound descriptions using both reference and imitation file names in the {RED}style of reference{RESET} descriptions."
    )

    # Build a dataset
    PromptBuilder = FilenameKeywordPromptBuilder()

    class FileNameDataset(Dataset):

        def __init__(self, files):
            self.files = files

        def __len__(self):
            return len(self.files)

        def __getitem__(self, index):
            filename = os.path.basename(self.files[index])
            return filename, PromptBuilder.generate_messages(filename)

    ds = FileNameDataset(audiofiles)
    dl = DataLoader(ds,
                    config.batch_size,
                    False,
                    num_workers=config.num_workers,
                    drop_last=False,
                    collate_fn=lambda batch: ([item[0] for item in batch], [item[1] for item in batch]))

    # Generate sound descriptions
    descriptions = {}
    for i, (filenames, messages_batch) in enumerate(tqdm(dl)):
        output_batch = pipe(messages_batch, batch_size=config.batch_size, **generation_args)
        for filename, output in zip(filenames, output_batch):
            descriptions[filename] = output[0]['generated_text'].split('|')
        print(f"Sample generated description for {YELLOW}{filename}{RESET}: {descriptions[filename]}")

    # Save the output to a JSON file
    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    with open(config.output_path, "w", encoding="utf-8") as f:
        json.dump(descriptions, f, ensure_ascii=False, indent=1)
    with open(config.output_path.replace('.json', '.log'), "w") as f:
        json.dump(vars(config), f, indent=2)
    print(f"Saved the generated descriptions to {RED}{config.output_path}{RESET}.")
    print(f"Saved logs to {RED}{config.output_path.replace('.json', '.log')}{RESET}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument parser for reference-style LLM-prompt generation on the VimSketch.")

    # General
    parser.add_argument('--num_workers', type=int, default=8, help="Number of data loader workers.")
    parser.add_argument('--dataset_path', type=str, default='data', help="Path to the data sets.")
    parser.add_argument('--output_path',
                        type=str,
                        default='',
                        help="Default: data/Vim_Sketch_Dataset/s2s/descriptions/references_[model]_[time].json")

    # LLM: microsoft/Phi-4, microsoft/Phi-4-mini-instruct have been tested
    parser.add_argument('--llm_dir', type=str, default='src/prepare_s2s_data/LLM', help="Path to the vLLM cache directory.")
    parser.add_argument('--llm_model', type=str, default='microsoft/Phi-4', help="LLM model name.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for LLM generation.")
    parser.add_argument('--max_new_tokens', type=int, default=600, help="Maximum number of new tokens to generate.")
    parser.add_argument('--return_full_text', type=bool, default=False, help="Whether to return including prompt.")
    parser.add_argument('--temperature', type=float, default=0.65, help="Sampling temperature for generation.")
    parser.add_argument('--top_p', type=float, default=0.9, help="Top-p sampling for generation.")
    parser.add_argument('--do_sample', default=True, action='store_true', help="Whether to use sampling.")
    parser.add_argument('--low_bit', default=False, action='store_true', help="Whether to use 4-bit quant.")
    args = parser.parse_args()

    if not args.output_path:
        model_short_name = args.llm_model.split('/')[-1]
        timestamp = datetime.now().strftime("%y%m%d%H%M")
        args.output_path = f'data/Vim_Sketch_Dataset/s2s/descriptions/references_{model_short_name}_{timestamp}.json'

    main(args)
