import json
import sys

result_arr = []
for i, filename in enumerate(sys.argv[1:]):
    with open(filename, 'r') as f:
        json_list = json.load(f)

        for json_obj in json_list:
            json_obj['tid'] = i
            result_arr.append(json_obj)

            
with open('output.json', 'w') as outf:
    json.dump(result_arr, outf, indent=4)