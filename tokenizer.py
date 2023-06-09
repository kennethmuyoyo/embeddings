# from tiktoken.models import Tokenizer
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def num_tokens(text: str) -> int:
    """Return the number of tokens in a string."""
    return len(tokenizer.encode(text))


def halved_by_delimiter(string: str, delimiter: str = "\n") -> list:
    """Split a string in two, on a delimiter, trying to balance tokens on each side."""
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]  # no delimiter found
    elif len(chunks) == 2:
        return chunks  # no need to search for halfway point
    else:
        total_tokens = num_tokens(string)
        halfway = total_tokens // 2
        best_diff = halfway
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)
            if diff >= best_diff:
                break
            else:
                best_diff = diff
        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        return [left, right]


def truncated_string(
    string: str,
    max_tokens: int,
    print_warning: bool = True,
) -> str:
    """Truncate a string to a maximum number of tokens."""
    encoded_string = tokenizer.encode(string)
    truncated_string = "".join(tokenizer.decode(encoded_string[:max_tokens]))
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning: Truncated string from {len(encoded_string)} tokens to {max_tokens} tokens.")
    return truncated_string



def split_strings_from_subsection(
    subsection: tuple,
    max_tokens: int = 1000,
    max_recursion: int = 5,
) -> list:
    """
    Split a subsection into a list of subsections, each with no more than max_tokens.
    Each subsection is a tuple of parent titles [H1, H2, ...] and text (str).
    """
    titles, text = subsection
    string = "\n\n".join(titles + [text])
    num_tokens_in_string = num_tokens(string)
    # if length is fine, return string
    if num_tokens_in_string <= max_tokens:
        return [string]
    # if recursion hasn't found a split after X iterations, just truncate
    elif max_recursion == 0:
        return [truncated_string(string, max_tokens=max_tokens)]
    # otherwise, split in half and recurse
    else:
        titles, text = subsection
        for delimiter in ["\n\n", "\n", ". "]:
            left, right = halved_by_delimiter(text, delimiter=delimiter)
            if left == "" or right == "":
                # if either half is empty, retry with a more fine-grained delimiter
                continue
            else:
                # recurse on each half
                results = []
                for half in [left, right]:
                    half_subsection = (titles, half)
                    half_strings = split_strings_from_subsection(
                        half_subsection,
                        max_tokens=max_tokens,
                        max_recursion=max_recursion - 1,
                    )
                    results.extend(half_strings)
                return results
    # otherwise no split was found, so just truncate (should be very rare)
    return [truncated_string(string, max_tokens=max_tokens)]

# split sections into chunks
MAX_TOKENS = 1600
content_strings = []

# Read the content from your file
with open("single_chunk.txt", "r") as file:
    content = file.read()

# Since your content doesn't have any sections or subsections, we treat the whole content as a single section
content_section = ([], content)

# Split the section into chunks
content_strings.extend(split_strings_from_subsection(content_section, max_tokens=MAX_TOKENS))

print(f"Content split into {len(content_strings)} strings.")

# Save each chunk to a separate file
for i, chunk in enumerate(content_strings):
    with open(f"chunk_{i}.txt", "w") as file:
        file.write(chunk)
