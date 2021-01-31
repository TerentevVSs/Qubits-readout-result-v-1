import csv
import random

"""
Программа для сбора полученных данных в один csv-файл.
"""


with open("0_new.txt", "r") as file:
    a = file.readlines()
    zeros = []
    label = []
    for elem in a[1:]:
        zeros.append(complex(elem.strip()))
        label.append(0)
with open("1_new.txt", "r") as file:
    a = file.readlines()
    ones = []
    for elem in a[1:]:
        ones.append(complex(elem.strip()))
        label.append(1)
data = zeros + ones
with open("Training examples4.csv", "a") as file:
    file_writer = csv.writer(file, delimiter=",", lineterminator="\r")
    for i in range(len(data)):
        file_writer.writerow([data[i].real, data[i].imag, label[i]])
