# import json
#
# with open(f"/home/data/ssh5131/focus_modeling/for_refiner_v2/wow_v2/train.json", 'r') as read_file:
#     wow_v2 = json.load(read_file)
#
#
# with open(f"/home/data/ssh5131/focus_modeling/for_refiner_v2/wow/new_train.json", 'r') as read_file:
#     wow_v1 = json.load(read_file)
#
#
# print(wow_v1)
# print()
# print(wow_v2)

from sklearn.utils import shuffle
list_1 = [1,2,3,4,5]
list_2 = [1,2,3,4,5]
list_1, list_2 = shuffle(list_1, list_2)

print(list_1)
print(list_2)