from tqdm import tqdm
import threading
from queue import Queue
from typing import Dict, Set, List, Tuple
import os
from collections import deque, defaultdict
import time
import sys
import re # Kept for word cleaning in auto_chain_queries

KB_limit = -1
BUFFER_SIZE = 999999

# translation_dict and categories remain the same
categories = ["what", "how"]
translation_dict = {
    "what": "descriptions.txt",  # nouns (can be subjects or objects)
    "how": "actions.txt",        # adverbs
    "do": "verbs.txt",           # verbs
    "describe": "picturable.txt", # articles/determiners
    "grade": "adj.txt",          # adjectives
    "form": "prep.txt"           # prepositions
}
list_of_words = []

# Removed SerialMonitor class

class SVOPattern:
    def __init__(self):
        self.subjects = defaultdict(set)      # subject -> verb
        self.verbs = defaultdict(set)         # verb -> object
        self.objects = defaultdict(set)       # object -> subject
        self.subject_object = defaultdict(set) # subject -> object

    def add_pattern(self, subject: str, verb: str, obj: str):
        self.subjects[subject].add(verb)
        self.verbs[verb].add(obj)
        self.objects[obj].add(subject)
        self.subject_object[subject].add(obj)

    def get_verbs_for_subject(self, subject: str) -> Set[str]:
        return self.subjects[subject]

    def get_objects_for_verb(self, verb: str) -> Set[str]:
        return self.verbs[verb]

    def get_subjects_for_object(self, obj: str) -> Set[str]:
        return self.objects[obj]

    def save_to_file(self, filename: str):
        with open(filename, "w", encoding="utf-8") as f:
            for subject in self.subjects:
                verbs = self.subjects[subject]
                for verb in verbs:
                    objects = self.verbs[verb]
                    for obj in objects:
                        f.write(f"{subject} {verb} {obj}\n")

    @classmethod
    def load_from_file(cls, filename: str) -> 'SVOPattern':
        pattern = cls()
        try:
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        subject, verb, obj = parts
                        pattern.add_pattern(subject, verb, obj)
            return pattern
        except FileNotFoundError:
            print(f"File {filename} not found.")
            return None
        except Exception as e:
            print(f"Error loading patterns: {e}")
            return None

class VocabularyCache:
    def __init__(self, translation_dict: Dict[str, str]):
        self.vocab_cache: Dict[str, Set[str]] = {}
        self._load_vocabularies(translation_dict)

    def _load_vocabularies(self, translation_dict: Dict[str, str]) -> None:
        for category, filename in translation_dict.items():
            with open(filename, 'r', encoding='utf-8') as f:
                self.vocab_cache[category] = {line.strip() for line in f.readlines()}

    def get_vocabulary(self, category: str) -> Set[str]:
        return self.vocab_cache.get(category, set())

    def is_word_in_category(self, word: str, category: str) -> bool:
        """Check if a word belongs to a specific category."""
        return word in self.vocab_cache.get(category, set())

    def find_word_category(self, word: str) -> str:
        """Find which category a word belongs to."""
        for category, words in self.vocab_cache.items():
            if word in words:
                return category
        return None

def process_sentence(sentence: str, vocab_cache: VocabularyCache, svo_patterns: SVOPattern = None) -> str:
    words = sentence.split()
    temp = "["

    # First pass: categorize words and track positions
    word_categories = {}
    for i, word in enumerate(words):
        for category, vocab in vocab_cache.vocab_cache.items():
            if word in vocab:
                word_categories[i] = (word, category)
                temp += f":{category}>{word}"

    # Second pass: identify SVO patterns
    if svo_patterns is not None:
        for i in range(len(words)-2):
            if i in word_categories and i+1 in word_categories and i+2 in word_categories:
                word1, cat1 = word_categories[i]
                word2, cat2 = word_categories[i+1]
                word3, cat3 = word_categories[i+2]

                # Check for SVO pattern
                if cat1 == "what" and cat2 == "do" and cat3 == "what":
                    svo_patterns.add_pattern(word1, word2, word3)

    temp += ":]\n"
    return temp if len(temp) > 3 else ""

import random

def generate_svo_sentence(svo_patterns: SVOPattern, vocab_cache: VocabularyCache, randomize: bool = False) -> str:
    if randomize:
        # Get all subjects that have associated verbs
        valid_subjects = [subj for subj in svo_patterns.subjects.keys() if svo_patterns.subjects[subj]]
        if not valid_subjects:
            return None

        # Randomly select a subject
        subject = random.choice(valid_subjects)

        # Get verbs associated with this subject and randomly select one
        possible_verbs = list(svo_patterns.get_verbs_for_subject(subject))
        if not possible_verbs: # Ensure there are verbs to choose from
            return None
        verb = random.choice(possible_verbs)

        # Get objects associated with this verb and randomly select one
        possible_objects = list(svo_patterns.get_objects_for_verb(verb))
        if not possible_objects: # Ensure there are objects to choose from
            return None
        obj = random.choice(possible_objects)

        return f"{subject} {verb} {obj}."
    else:
        # Pattern-based SVO generation
        for subject in svo_patterns.subjects:
            verbs_for_subject = svo_patterns.get_verbs_for_subject(subject)
            if verbs_for_subject:
                for verb in verbs_for_subject:
                    objects = svo_patterns.get_objects_for_verb(verb)
                    if objects:
                        obj = next(iter(objects))
                        return f"{subject} {verb} {obj}."
    return None


def print_word_by_word(sentence: str, delay: float = 1.0) -> tuple:
    """
    Print a sentence one word at a time with a delay between words.
    Returns (list_of_words, completed_flag)
    """
    global list_of_words # Using global list_of_words
    list_of_words.clear() # Clear for this sentence

    if not sentence:
        return list_of_words, True

    words = sentence.split()
    for i, word in enumerate(words):
        list_of_words.append(word)
        #sys.stdout.write(word)
        #sys.stdout.flush()

        # Add space after word (except for last word and punctuation)
        #if i < len(words) - 1 and not words[i+1] in ['.', ',', '!', '?', ';', ':']:
            #sys.stdout.write(' ')
            #sys.stdout.flush()

        # Delay between words

    # Print newline at the end
    # sys.stdout.write('\n') # Consider if newline is desired here or after the call
    # sys.stdout.flush()
    return list_of_words, True # Always completes successfully without serial interruption

class ResultBuffer:
    def __init__(self, output_file: str, buffer_size: int = BUFFER_SIZE):
        self.output_file = output_file
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.lock = threading.Lock()
        self.flush_count = 0

    def add_result(self, result: str) -> None:
        with self.lock:
            self.buffer.append(result)
            if len(self.buffer) >= self.buffer_size:
                self.flush_buffer()

    def flush_buffer(self) -> None:
        if not self.buffer:
            return

        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                while self.buffer:
                    f.write(self.buffer.popleft())
            self.flush_count += 1
        except Exception as e:
            print(f"Error writing to file: {e}")
            # Re-add to buffer if write failed, though this could lead to an infinite loop
            # if the file is permanently unwritable. Consider more robust error handling.
            # For now, just printing error. If you want to re-add, ensure to handle order.
            # self.buffer.extendleft(reversed(list_to_re_add))) # if items were popped

    def final_flush(self) -> None:
        with self.lock:
            self.flush_buffer()

def worker(sentence_queue: Queue, result_buffer: ResultBuffer,
           vocab_cache: VocabularyCache, svo_patterns: SVOPattern,
           pbar: tqdm) -> None:
    while True:
        try:
            sentence = sentence_queue.get_nowait()
        except Queue.Empty: # Corrected from Queue.empty to Queue.Empty
            break

        if sentence is None: # Sentinel value to stop worker
            break

        result = process_sentence(sentence, vocab_cache, svo_patterns)
        if result:
            result_buffer.add_result(result)
        pbar.update(1)
        sentence_queue.task_done()

def build_memory_multithreaded(filename: str, num_threads: int = None) -> SVOPattern:
    if num_threads is None:
        num_threads = os.cpu_count() or 4

    print(f"\nBuilding memory using {num_threads} threads...")

    sentence_queue = Queue()
    result_buffer = ResultBuffer("memory.txt")
    svo_patterns = SVOPattern() # This will be populated by workers
    vocab_cache = VocabularyCache(translation_dict)

    try:
        with open(filename, 'r', encoding='latin-1') as f: # latin-1 might be problematic, utf-8 is safer
            text_content = f.read()
            # Limit words if KB_limit is set and positive
            if KB_limit > -1:
                 words_for_limit = text_content.split()[:KB_limit]
                 text = ' '.join(words_for_limit)
            else:
                 text = ' '.join(text_content.split()) # Normalize whitespace

        sentences = [s.strip() for s in text.split(".") if s.strip()]
    except FileNotFoundError:
        print(f"Error: Training file '{filename}' not found.")
        return svo_patterns # Return empty patterns
    except Exception as e:
        print(f"Error reading training file '{filename}': {e}")
        return svo_patterns # Return empty patterns


    for sentence in sentences:
        sentence_queue.put(sentence)

    # Add sentinel values for each thread to stop
    for _ in range(num_threads):
        sentence_queue.put(None)

    pbar = tqdm(total=len(sentences), desc="Processing Sentences", unit="sentence")

    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(
            target=worker,
            args=(sentence_queue, result_buffer, vocab_cache, svo_patterns, pbar)
        )
        thread.daemon = True # Allow main program to exit even if threads are running
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join() # Wait for all threads to complete

    result_buffer.final_flush()
    pbar.close()

    print(f"\nMemory building complete. Buffer flushed {result_buffer.flush_count} times.")
    return svo_patterns


def print_query_results_word_by_word(results: set, delay: float = 1.0) -> tuple:
    """
    Print query results one word at a time.
    Returns (completed_flag, last_word)
    """
    if not results:
        print("[ No results found ]")
        return True, None

    words = list(results) # Convert set to list to ensure order for printing if needed
    results_str = "[ " + ' '.join(words) + " ]"
    
    # print_word_by_word now returns (list_of_printed_words, completed_flag)
    # We are interested in the last word from the `words` list, not `list_of_words` global
    # The `completed` flag will always be True now.
    printed_words_list, completed = print_word_by_word(results_str, delay)
    #sys.stdout.write('\n') # Add newline after printing all words
    sys.stdout.flush()

    if printed_words_list: # Check if anything was actually printed
        # The last word of the *original* results set (now a list)
        # that was intended to be printed
        # However, print_word_by_word appends to a global list_of_words
        # which also contains the brackets. We want the last *actual* word from `words`.
        if words:
             # Return the last word from the *input* 'words' list if it exists
             # and the printing process was considered completed.
            return completed, random.choice(words) # Return the last word of the original query results
    return completed, None


# Removed setup_serial_connection function

def auto_chain_queries(vocab_cache: VocabularyCache, word_delay: float, num_iterations=10000, initial_category="what", initial_word=None):
    """
    Run a series of queries that automatically chain from one to the next.
    Each query uses the last result from the previous query, alternating categories.
    """
    global list_of_words # Access the global list_of_words
    current_category_idx = categories.index(initial_category) if initial_category in categories else 0

    current_word = initial_word
    if not current_word:
        category_vocab = vocab_cache.get_vocabulary(categories[current_category_idx])
        if category_vocab:
            svo_patterns = SVOPattern.load_from_file("SVO.txt")
            if svo_patterns:
                # Ensure generate_svo_sentence returns a valid string before splitting
                generated_sentence = generate_svo_sentence(svo_patterns, vocab_cache, randomize=True)
                if generated_sentence:
                    rand_words = generated_sentence.split()
                    if rand_words:
                        current_word = random.choice(list(rand_words))
                        current_word = re.sub(r'[^\w\s]', '', current_word) # Clean punctuation
                        with open("output.txt", "a", encoding="utf-8") as file:
                            file.write("Starting with: " + current_word + "\n")
                    else:
                        print("Could not generate random words for starting.")
                        return
                else:
                    print(f"Could not generate an SVO sentence to pick a random word from category '{categories[current_category_idx]}'.")
                    return
            else:
                print("Could not load SVO.txt to generate a random starting word.")
                return

        else:
            print(f"No words found in category '{categories[current_category_idx]}'")
            return
    if not current_word: # If still no current_word after trying to find one
        print("Failed to determine a starting word. Stopping auto-chain.")
        return


    print(f"\nStarting auto-chain with: {categories[current_category_idx]} {current_word}")
    print(f"Will run for {num_iterations} iterations in pattern: {' → '.join(categories)} → ...")
    time.sleep(2)

    try:
        with open("memory.txt", "r", encoding="utf-8") as f:
            data = f.readlines()
    except FileNotFoundError:
        print("Error: memory.txt not found. Please build memory first (option 1).")
        return

    chain_history = [(categories[current_category_idx], current_word)]

    for i in range(num_iterations):
        if not current_word:
            print("\nNo word to continue with. Stopping auto-chain.")
            break

        current_category_name = categories[current_category_idx] # Actual name of category
        #print(f"\nIteration {i+1}/{num_iterations}: Querying for words related to '{current_word}' (from category '{current_category_name}')")
        
        next_category_idx = (current_category_idx + 1) % len(categories)
        next_category_name = categories[next_category_idx]
        #print(f"Looking for results in category: '{next_category_name}'")


        out = set()
        # No serial interruption logic needed here

        for sub_text in data:
            lingual = sub_text.split(":")
            words_in_entry = defaultdict(set) # Use defaultdict for convenience

            for group in lingual:
                parts = group.split(">")
                if len(parts) > 1:
                    element0 = parts[0].strip()
                    element1 = parts[1].strip().replace(']', '') # Clean trailing bracket if any
                    if element0 and element1:
                        words_in_entry[element0].add(element1)
            
            # Define relationship mappings (source_category -> target_category)
            # We are looking for entries where current_word is in a source_category
            # and we want to extract words from the associated target_category IF
            # that target_category is our desired next_category_name.
            relationship_mappings = [
                ("what", "do"), ("how", "do"), ("describe", "what"),
                ("grade", "what"), ("what", "how"), ("describe", "grade"),
                ("how", "what"), ("form", "what"), ("form", "describe"),
                ("form", "grade"), ("form", "how")
            ]
            
            found_current_word_in_source = False
            # Check if current_word exists in any of the source categories of the current entry
            for source_cat, target_cat in relationship_mappings:
                if current_word in words_in_entry.get(source_cat, set()):
                    found_current_word_in_source = True
                    # If this relation's target is what we're looking for, add its words
                    if target_cat == next_category_name:
                         out.update(words_in_entry.get(target_cat, set()))
            
            # Fallback: If current_word was found in *any* source category in this entry,
            # but no specific relation led to 'next_category_name',
            # then add all words from 'next_category_name' found in this entry.
            # This part might be too broad, consider if this is desired.
            # For now, sticking to the more direct relation.
            # A simpler approach if the above logic isn't yielding results:
            # If current_word is found in *any* category C1 in the entry,
            # and the entry also contains words for next_category_name (C2), add them.
            # This is what the original logic seemed to imply more broadly.
            # Let's refine: We only care if current_word appears as current_category_name
            # or if it appears as a source that *could* lead to next_category_name.
            
            # Simpler logic attempt based on original structure:
            # If the current_word is found under *any* category key in words_in_entry
            # and words_in_entry also has the next_category_name key, add those.
            # This is too broad.
            
            # Revised logic based on intent:
            # The current_word is associated with current_category_name.
            # We need to find entries where current_word appears under *some* category (source_cat_in_entry)
            # and that same entry has words for our desired next_category_name (target_cat_in_entry).
            
            # Sticking to the clearer relationship_mappings logic.
            # The goal is: if current_word (known to be of current_category_name)
            # is found in a 'source' part of a relationship, collect from the 'target' part
            # *if* that 'target' part matches 'next_category_name'.

        # Filter results to only include words from the next_category_name
        # This step is crucial because 'out' might have collected words from other categories
        # if the relationship_mappings logic was broader.
        # With the current precise mapping, this might be redundant but safe.
        final_results_for_next_step = set()
        for word_candidate in out:
            # We must ensure the candidate word is actually in the vocabulary for the next_category_name
            if vocab_cache.is_word_in_category(word_candidate, next_category_name):
                final_results_for_next_step.add(word_candidate)
        
        if not final_results_for_next_step and out: # If filtering removed everything, but we had *some* related words
            # This fallback might be too broad, but it's an option if direct category matches are sparse.
            # Try to pick any word from 'out' that is in *any* category and see if that helps restart a chain.
            # For now, let's be strict: if no words for next_category_name, it's a failed step.
            pass # print(f"Found related words, but none in the target category '{next_category_name}'.")


        # print(f"\nPotential next words for '{next_category_name}' related to '{current_word}':")
        # Use final_results_for_next_step for printing
        # The print_query_results_word_by_word will print the [ brackets ]
        # We need to make sure current_word has the correct context for the printout
        print(f"\nFound for '{current_word}' (as '{current_category_name}') -> looking for '{next_category_name}':")

        completed, last_word_printed = print_query_results_word_by_word(final_results_for_next_step, word_delay)
        
        # last_word_printed is the actual last word from the results that was shown.
        # It should belong to next_category_name if filtering worked.

        previous_word_for_log = current_word
        current_word = last_word_printed # This will be None if no results or print was "interrupted" (not possible now)

        with open("output.txt", "a", encoding="utf-8") as file:
            if current_word:
                file.write(f"{previous_word_for_log} ({current_category_name}) -> {current_word} ({next_category_name})\n")
            else:
                file.write(f"{previous_word_for_log} ({current_category_name}) -> NO RESULT for {next_category_name}\n")

        if current_word:
            chain_history.append((next_category_name, current_word))
            current_category_idx = next_category_idx # Successfully moved to the next category
        else:
            # No valid result, try to pick a random word from the *next intended* category
            print(f"\nNo results found for {next_category_name} from {previous_word_for_log}. Attempting to restart with random word from {next_category_name}.")
            current_category_idx = next_category_idx # Advance category index anyway for the random pick
            
            category_vocab = vocab_cache.get_vocabulary(categories[current_category_idx])
            if category_vocab:
                svo_patterns_local = SVOPattern.load_from_file("SVO.txt") # Re-load or pass if already loaded
                if svo_patterns_local:
                    generated_sentence = generate_svo_sentence(svo_patterns_local, vocab_cache, randomize=True)
                    if generated_sentence:
                        rand_words = generated_sentence.split()
                        # Filter these random words to only those belonging to the current target category
                        valid_random_words_for_category = [w for w in rand_words if vocab_cache.is_word_in_category(re.sub(r'[^\w\s]', '', w), categories[current_category_idx])]
                        if valid_random_words_for_category:
                            current_word = random.choice(valid_random_words_for_category)
                            current_word = re.sub(r'[^\w\s]', '', current_word) # Clean it
                            print(f"Randomly selected new word: {current_word} (for category {categories[current_category_idx]})")
                            with open("output.txt", "a", encoding="utf-8") as file:
                                file.write(f"RESTART with random word: {current_word} ({categories[current_category_idx]})\n")
                            chain_history.append((categories[current_category_idx], current_word))
                        else:
                            print(f"Could not find a random word for category {categories[current_category_idx]} from SVO sentence. Stopping chain.")
                            break 
                    else:
                        print(f"Could not generate SVO sentence for random word. Stopping chain.")
                        break
                else:
                    print("Could not load SVO.txt for random word. Stopping chain.")
                    break
            else:
                print(f"No vocabulary for category {categories[current_category_idx]} to pick a random word. Stopping chain.")
                break
        time.sleep(1)

    with open("output.txt", "a", encoding="utf-8") as file:
        file.write("\n--- CHAIN SUMMARY ---\n")
        for idx, (cat, word) in enumerate(chain_history):
            file.write(f"{idx+1}. {cat}: {word}\n")

    print("\nAuto-chain complete. Full results saved to output.txt")


def main():
    print(translation_dict)
    svo_patterns = None # Will be loaded or built
    vocab_cache = None

    word_delay = 0.7

    try:
        vocab_cache = VocabularyCache(translation_dict)
    except FileNotFoundError as e:
        print(f"Warning: Could not load vocabulary files: {e}")
        print("Make sure all required vocabulary files (e.g., descriptions.txt, actions.txt) exist in the current directory.")
        print("Proceeding without preloaded vocabulary. Some functions might fail until vocabulary is loaded (e.g. via building memory).")
    except Exception as e:
        print(f"An unexpected error occurred while loading vocabulary: {e}")


    while True:
        print("\nOptions:")
        print("1. Build memory")
        print("2. Execute queries")
        print("3. Auto-chain queries (what-how pattern)") # Was 5
        print("4. Exit") # Was 6

        choice = input("\nEnter your choice (1-5): ").strip()

        if not vocab_cache: # Attempt to load vocab again if it failed initially and user selected an option that needs it
            try:
                vocab_cache = VocabularyCache(translation_dict)
            except Exception as e:
                # If trying to build memory, vocab_cache is created inside build_memory_multithreaded
                if choice != "1": 
                    print(f"Error loading vocabulary: {e}. Please ensure vocabulary files are present.")
                    print("You might need to run 'Build memory' first if files are missing, as it also loads vocabulary.")
                    if choice != "5": # Don't continue if exiting
                        continue 


        if choice == "1":
            filename = input("Enter training file path: ")
            if not os.path.exists(filename):
                print(f"Error: Training file '{filename}' not found.")
                continue
            num_threads_str = input("Enter number of threads (press Enter for auto): ").strip()
            num_threads = int(num_threads_str) if num_threads_str else None
            
            # build_memory_multithreaded creates its own vocab_cache internally if needed,
            # but it's better if the main one is already loaded.
            # It also returns svo_patterns.
            svo_patterns = build_memory_multithreaded(filename, num_threads)
            if svo_patterns: # Check if svo_patterns is not None
                 svo_patterns.save_to_file("SVO.txt")
                 print("SVO patterns saved to SVO.txt")
            else:
                print("Memory building failed or returned no patterns.")
            # Ensure main vocab_cache is up-to-date if build_memory loaded it
            if not vocab_cache:
                try:
                    vocab_cache = VocabularyCache(translation_dict)
                except Exception as e:
                    print(f"Failed to reload main vocabulary cache after memory build: {e}")


        elif choice == "2":
            if not vocab_cache:
                print("Vocabulary not loaded. Please build memory (option 1) or ensure vocab files are present.")
                continue
            while True:
                query_input = input("Enter command (e.g., 'what word1 word2' or 'back' to return): ").split()
                if not query_input or query_input[0].lower() == 'back':
                    break

                if len(query_input) < 2:
                    print("Please enter a category (e.g., 'what') and at least one word (e.g., 'helped').")
                    continue

                category_to_search_for_others = query_input[0] # The category of words we want to find
                search_words_as_input = query_input[1:] # The words we are using as input for the search

                # Determine the category of the input search words (simplified: assume all same, or handle complex later)
                # For now, this query type doesn't strictly use the input words' category in its logic,
                # it just searches for lines containing the input words and extracts related words based on mappings.
                # The 'category_to_search_for_others' is not directly used in the current search logic below,
                # rather the relationship_mappings dictate what is extracted.
                # The prompt structure "category word" implied finding other words related to "word"
                # where the *output* words might belong to "category".
                # This part needs clarification if the logic isn't matching intent.
                # The current loop below finds words associated via relationship_mappings,
                # not necessarily words of 'category_to_search_for_others'.

                out = set()
                try:
                    with open("memory.txt", "r", encoding="utf-8") as f:
                        data = f.readlines()

                    with tqdm(total=len(data), desc="Processing data", unit="segment") as pbar:
                        for sub_text in data:
                            pbar.update(1)
                            lingual = sub_text.split(":")
                            words_in_entry = defaultdict(set)

                            for group in lingual:
                                parts = group.split(">")
                                if len(parts) > 1:
                                    element0 = parts[0].strip()
                                    element1 = parts[1].strip().replace(']', '') # Clean
                                    if element0 and element1:
                                        words_in_entry[element0].add(element1)
                            
                            # Check if *any* of the search_words_as_input are present in this entry's sources
                            found_any_search_word = False
                            for search_word in search_words_as_input:
                                for source_cat_key in words_in_entry:
                                    if search_word in words_in_entry[source_cat_key]:
                                        found_any_search_word = True
                                        break
                                if found_any_search_word:
                                    break
                            
                            if found_any_search_word:
                                relationship_mappings = [
                                    ("what", "do"), ("how", "do"), ("describe", "what"),
                                    ("grade", "what"), ("what", "how"), ("describe", "grade"),
                                    ("how", "what"), ("form", "what"), ("form", "describe"),
                                    ("form", "grade"), ("form", "how")
                                ]
                                for search_word_single in search_words_as_input: # Iterate through each input word
                                    for source_cat, target_cat in relationship_mappings:
                                        # If the current search_word_single is in the source_cat of the entry
                                        if search_word_single in words_in_entry.get(source_cat, set()):
                                            # Add all words from the corresponding target_cat
                                            out.update(words_in_entry.get(target_cat, set()))
                                            # Optionally, filter 'out' to only words of 'category_to_search_for_others'
                                            # For now, it collects all related words as per mappings.

                except FileNotFoundError:
                    print("Error: memory.txt not found. Please build memory first (option 1).")
                    continue
                
                # Optionally filter 'out' to only words belonging to 'category_to_search_for_others'
                final_output = set()
                if category_to_search_for_others in translation_dict: # valid category name
                    for word in out:
                        if vocab_cache.is_word_in_category(word, category_to_search_for_others):
                            final_output.add(word)
                    if not final_output and out: # If filtering removed everything but there were initial results
                         print(f"(Note: Found related words, but none matched the specific category '{category_to_search_for_others}'. Showing all found.)")
                         final_output = out # Show all if specific category yields nothing
                else:
                    print(f"Warning: Category '{category_to_search_for_others}' is not a known queryable category. Showing all related words.")
                    final_output = out


                print_query_results_word_by_word(final_output, word_delay)


        elif choice == "3": # Was 5 (Auto-chain)
            if not vocab_cache:
                print("Vocabulary not loaded. Please build memory (option 1) or ensure vocab files are present.")
                continue
            if not os.path.exists("memory.txt"):
                print("Error: memory.txt not found. Please build memory first (option 1).")
                continue
            if not os.path.exists("SVO.txt"):
                print("Error: SVO.txt not found. Please build memory first (option 1) to generate SVO patterns.")
                continue


            print("\nAuto-Chain Query Configuration")
            try:
                iterations_str = input("Enter number of iterations to run (default 10000): ")
                iterations = int(iterations_str) if iterations_str else 10000
            except ValueError:
                print("Invalid input. Using default value of 10000 iterations.")
                iterations = 10000

            start_category = input(f"Enter starting category ({', '.join(categories)}, default '{categories[0]}'): ").lower() or categories[0]
            if start_category not in categories:
                print(f"Invalid category '{start_category}'. Using '{categories[0]}' instead.")
                start_category = categories[0]

            start_word = input(f"Enter starting word for '{start_category}' (or leave empty for random): ").strip()
            if start_word and not vocab_cache.is_word_in_category(start_word, start_category):
                # Allow if user insists, but warn
                print(f"Warning: '{start_word}' is not officially in category '{start_category}' per vocab files. Will use it if you proceed.")
                confirm = input(f"Continue with '{start_word}' as '{start_category}'? (y/n): ").lower()
                if confirm != 'y':
                    start_word = None # Reset to get random if user aborts
            if not start_word: # If empty or user aborted above
                print(f"No specific start word provided for {start_category}, will select one randomly.")


            auto_chain_queries(
                vocab_cache=vocab_cache,
                word_delay=word_delay,
                num_iterations=iterations,
                initial_category=start_category,
                initial_word=start_word
            )

        elif choice == "4": # Was 6 (Exit)
            # No serial monitor to disconnect
            print("Exiting program...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
      main()