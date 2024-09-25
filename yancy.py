import collections

def decode(message_file):
    # Basically we read the file first, put num: word to a dict
    # Also we find the largest num of the file
    # Then we can find the end num of line 1 is 1, 1 + 2 reaches the end of the second line 3,
    # 3 + 3 reaches the end of the third line, ...
    # which means i + factor reaches the end of each line till the largest num, and factor increases 1 during each loop
    # So we get all the end num of each line, and we find the correspond word in the dict, and put them in the ans.
    keys, key2Word, largest, ans = set(), {}, 0, []
    # Read the txt file
    with open(message_file, "r") as file:
        for line in file:
            line = line.split()
            key, word = int(line[0]), line[1]
            key2Word[key] = word
            largest = max(largest, key)
    # Go through each end of line and put them in ans
    i, factor = 1, 2
    while i <= largest:
        ans.append(key2Word[i])
        i, factor = i + factor, factor + 1
    return " ".join(ans)


def SS(ls):
    for i in range(len(ls) - 1):
        mini = ls[i]
        for j in range(i + 1, len(ls)):
            if ls[j] < mini:
                mini = ls[j]

        ls[i], mini = mini, ls[i]
    print(ls)


if __name__ == "__main__":
    # print(decode('coding_qual_input.txt'))
    SS([1, 2, 3, 5, 6, 8, 5, 3, 2, 1, 3, 7])