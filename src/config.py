import os

big_letters = list(range(ord('A'), ord('Z')+1))
digits = list(range(ord('0'), ord('9')+1))
separator = ord("-")
alphabet = big_letters + digits
alphabet.append(separator)
absolute_max_string_len = 4 + 1 + 4
image_size = (200,50)
image_chanels_size = (200,50,3)
plate_min = 3
plate_max = 4

epochs = 20
steps_per_epoch = 256
minbatch = 64



font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'resources', 'Assistant-Regular.ttf')
