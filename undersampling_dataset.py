import numpy as np

all_input = np.load('feature_policy2d.npy')
all_label = np.load('label_policy2d.npy')

input_pos = all_input[all_label > 0]
input_neg = all_input[all_label == 0]
idx = np.arange(len(input_neg))
idx_choose = np.random.choice(idx, size=len(input_pos), replace=False)
input_neg_choosen = input_neg[idx_choose]

label_pos_new = np.ones(len(input_pos))
label_neg_new = np.zeros(len(input_neg_choosen))

print(np.shape(input_pos), np.shape(input_neg_choosen))

input_new = np.concatenate([input_pos, input_neg_choosen])
label_new = np.concatenate([label_pos_new, label_neg_new])

randomizer = np.arange(len(input_new))
np.random.shuffle(randomizer)
input_new = input_new[randomizer]
label_new = label_new[randomizer]
np.save('feature_policy2d_train.npy', input_new)
np.save('label_policy2d_train.npy', label_new)