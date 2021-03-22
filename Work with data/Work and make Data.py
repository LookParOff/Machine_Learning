import pandas as pd
import numpy as np
import re
fileInp = open("..//Datasets/private datasets/mesTatyana/messages before 02.21.txt", "r", encoding="UTF-8")
fileOut = open("..//Datasets/private datasets/mesTatyana/FIX messages before 02.21.txt", "w", encoding="UTF-8")

attachments = False
for line in fileInp:
    if len(line.strip()) == 0:
        attachments = False
        continue

    if re.search("[а-яА-ЯёЁ]{3,} [а-яА-ЯёЁ]{3,} \(\d{1,2} .* \d{1,2}:\d{2}:\d{2} .{2}\):", line) is not None:
        fileOut.write("\n" + line.strip() + " ")
    else:
        fileOut.write(line.strip() + " ")
print("end")
