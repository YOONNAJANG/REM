import json

with open(f"/home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v2/train.json", 'r') as read_file:
    wow_v2 = json.load(read_file)


with open(f"/home/data/ssh5131/focus_modeling/for_refiner_v2/wow/new_train.json", 'r') as read_file:
    wow_v1 = json.load(read_file)


print(wow_v1)
print()
print(wow_v2)