#encoding=utf8

import os

print("Generate digits test dataset...")

test_digits_folder = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'digits/test_digits'))
test_digits_file_path = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'digits_multiclass_test.dat')

for (dirpath, dirnames, filenames) in os.walk(test_digits_folder):
    test_digit_content = ''
    for file_name in filenames:
        test_file_path = os.path.normpath(os.path.join(dirpath, file_name))
        digit_class = file_name[0:1]
        one_line_string = ''
        with open(test_file_path) as f:
            for line in f:
                space_line = " ".join(line).rstrip() + " "
                one_line_string = one_line_string + space_line
        one_line_string = one_line_string + digit_class + "\n"
        test_digit_content = test_digit_content + one_line_string

    target_file = open(test_digits_file_path, 'w')
    target_file.write(test_digit_content)
    target_file.close()

print("Generate digits test complete.")

print("Generate digits training dataset...")

train_digits_folder = os.path.normpath(os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'digits/training_digits'))
train_digits_file_path = os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), 'digits_multiclass_train.dat')

for (dirpath, dirnames, filenames) in os.walk(train_digits_folder):
    train_digit_content = ''
    for file_name in filenames:
        train_file_path = os.path.normpath(os.path.join(dirpath, file_name))
        digit_class = file_name[0:1]
        one_line_string = ''
        with open(train_file_path) as f:
            for line in f:
                space_line = " ".join(line).rstrip() + " "
                one_line_string = one_line_string + space_line
        one_line_string = one_line_string + digit_class + "\n"
        train_digit_content = train_digit_content + one_line_string

    target_file = open(train_digits_file_path, 'w')
    target_file.write(train_digit_content)
    target_file.close()

print("Generate digits training complete.")
