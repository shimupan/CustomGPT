import os
import json

base_dir = './messages'

with open('output.txt', 'w', encoding='utf-8') as outfile:
    for root, dirs, files in os.walk(base_dir):
        if 'messages.json' in files:
            file_path = os.path.join(root, 'messages.json')
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                for row in data:
                    if 'Contents' in row:
                        outfile.write(row['Contents'] + '\n')