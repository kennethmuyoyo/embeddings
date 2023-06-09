import nltk
import os

nltk.download('punkt')

SECTIONS_TO_IGNORE = [
    # Add more part headers here
]

# def split_into_chunks(filename):
#     with open(filename, 'r') as file:
#         text = file.read()
#         # You can adjust the delimiter as needed, here it is set to split by section and subsection titles
#         chunks = nltk.tokenize.sent_tokenize(text)
#     return chunks

# def filter_chunks(chunks):
#     chunks = [chunk for chunk in chunks if not any(section in chunk for section in SECTIONS_TO_IGNORE)]
#     return chunks

# filename = "YT.txt"  # Adjust this to point towards your YT.txt file

# chunks = split_into_chunks(filename)
# filtered_chunks = filter_chunks(chunks)

# for i, chunk in enumerate(filtered_chunks):
#     with open(f'chunk_{i}.txt', 'w') as file:
#         file.write(chunk)

def save_as_single_chunk(filename):
    with open(filename, 'r') as file:
        text = file.read().replace('\n', ' ')
    with open('single_chunk.txt', 'w') as file:
        file.write(text)

filename = "YT.txt"  # Adjust this to point towards your YT.txt file

save_as_single_chunk(filename)
