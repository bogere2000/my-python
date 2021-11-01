import random

from words import words
import string

def get_valid_word(words):
      word = random.choice(words)
      while '_' in word or '' in words:
          word = random.choice(words)

      return word
def hangman():
 word = get_valid_word(words)
word_letters = set(words)
alphabet = set(string.ascii_uppercase)
used_letters = set()
lives = 6
while len(word_letters) >0 and lives >0:
    print('you have', lives, 'lives left and you have used the letters: ', ''.join(used_letters))
    word_list = [letter if letter in used_letters else '_' for letter in words]
    print('current word:',''.join(word_list))
    user_letter = input('Guess a letter').upper()
if user_letter in alphabet - used_letters:
    used_letters.add(user_letter)
    if user_letter in word_letters:
        word_letters.remove(user_letter)

    else:
        lives = lives-1
        print('letter is not in word')

elif user_letter in used_letters:
    print('you have already used that character.please try again')
else:
    print('Invalid character.try again')
    if lives == 0:
        print('you died, sorry. The word was', word)
    else:
        print('you guessed the word', word ,'!!')

hangman()
