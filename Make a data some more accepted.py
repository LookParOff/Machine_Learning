def isWordRussian(word):
    for symb in word:
        if symb not in RUSSIAN_ALFABET:
            return False
    return True


letters = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ".lower()
RUSSIAN_ALFABET = set(list(letters))


file = open("data/War and peace all toms.txt", "r", encoding="UTF-8")
fileWithImprovedData = open("data/Improved War and peace.txt", "w", encoding="UTF-8")

for line in file:
    if len(line.split()) == 0:
        continue
    for index in range(len(line.split())):
        word = line.split()[index].lower()
        if isWordRussian(word) and 1 < len(word) <= 10:
            fileWithImprovedData.write(word + " ")
    fileWithImprovedData.write("\n")

file.close()
fileWithImprovedData.close()
