import math


"""  first word (simplified)  """
#  Solution 01
# def first_word(text: str) -> str:
#     return text.split()[0]

#  Solution 02
# def first_word( text: str ) -> str:
#     j = 0
#     while j < len(text) and text[j] != " ":
#         j += 1
#     return text[:j]

#  Solution 03
first_word = lambda s:s.split()[0]


"""  Acceptable Password I  """
def is_acceptable_password( password: str ) -> bool:
    return len(password) > 6


"""  Replace First  """
#  Solution 01
# def replace_first( arr: list ) -> list:
#     return arr[1:] + arr[:1] if len(arr) > 1 else arr

#  Solution 02
replace_first = lambda x: x[1:] + x[:1]


"""  Split Pairs  """
def split_pairs( text:str ) -> list:
    text += "_" if len(text) % 2 else ""
    return [ text[i: i+2] for i in range(0, len(text), 2) ]


"""  Nearest Value  """
def nearest_value( nums: set, key: int) -> int:
    min_abs = float(math.inf)
    ans = float(math.inf)
    for num in nums:
        diff = abs(num - key)
        if diff < min_abs:
            min_abs = diff
            ans = num
        elif diff == min_abs:
            ans = min(ans, num)
    return ans


"""  Correct_Sentence  """
correct_sentence = lambda s: s[:1].upper() + s[1:] + "." * (s[-1] != ".")


"""  Is Even """
def is_even( num: int ) -> bool:
    return num & 1 == 0


"""  Sum Numbers  """
#  Solution 01
# sum_numbers = lambda text: sum( int(word) for word in text.split() if word.isdigit())

#  Solution 02
def sum_numbers( text: str ) -> int:
    #  注意：filter(函数名不带括号，且是str.， 循环体带括号)
    return sum(map(int, filter(str.isdigit, text.split())))


def checkio(text: str) -> bool:
    text = text.split()
    # i is the start pointer, j is the end pointer.
    i = -1
    for j in range(len(text)):
        if text[j].isdigit(): i = j
        elif j - i == 3: return True
    return False


if __name__ == '__main__':
    # These "asserts" are used for self-checking and not for an auto-testing
    assert first_word("Hello world") == "Hello"
    assert first_word("a word") == "a"
    assert first_word("hi") == "hi"

    assert is_acceptable_password('short') == False
    assert is_acceptable_password('muchlonger') == True

    assert replace_first([1, 2, 3, 4]) == [2, 3, 4, 1]
    assert replace_first([1]) == [1]

    assert split_pairs('abcd') == ['ab', 'cd']
    assert split_pairs('abc') == ['ab', 'c_']

    assert nearest_value({4, 7, 10, 11, 12, 17}, 9) == 10
    assert nearest_value({4, 7, 10, 11, 12, 17}, 8) == 7
    assert nearest_value({4, 8, 10, 11, 12, 17}, 9) == 8
    assert nearest_value({4, 9, 10, 11, 12, 17}, 9) == 9
    assert nearest_value({4, 7, 10, 11, 12, 17}, 0) == 4
    assert nearest_value({4, 7, 10, 11, 12, 17}, 100) == 17
    assert nearest_value({5, 10, 8, 12, 89, 100}, 7) == 8
    assert nearest_value({-1, 2, 3}, 0) == -1

    assert correct_sentence("greetings, friends") == "Greetings, friends."
    assert correct_sentence("Greetings, friends") == "Greetings, friends."
    assert correct_sentence("Greetings, friends.") == "Greetings, friends."
    assert correct_sentence("hi") == "Hi."
    assert correct_sentence("welcome to New York") == "Welcome to New York."

    assert is_even(2) == True
    assert is_even(5) == False
    assert is_even(0) == True

    assert sum_numbers('hi') == 0
    assert sum_numbers('who is 1st here') == 0
    assert sum_numbers('my numbers is 2') == 2
    assert sum_numbers('This picture is an oil on canvas '
                       'painting by Danish artist Anna '
                       'Petersen between 1845 and 1910 year') == 3755
    assert sum_numbers('5 plus 6 is') == 11
    assert sum_numbers('') == 0

    assert checkio("Hello World hello") == True, "Hello"
    assert checkio("He is 123 man") == False, "123 man"
    assert checkio("1 2 3 4") == False, "Digits"
    assert checkio("bla bla bla bla") == True, "Bla Bla"
    assert checkio("Hi") == False, "Hi"

    print("You did it!")