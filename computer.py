import random
def guess(x):
    random_number = random.randint(1, x)
    guess = 0
    while guess != random_number:
        guess = int(input(f'Guess a number between 1 and {x}:'))
        print(guess)
        if guess < random_number:
            print('sorry, guess again. too low')
        elif guess > random_number:
            print('sorry, guess again.too high')
        else:
            guess = random_number
            print(f'congrats, you have guessed the number')

def computer_guess(x):
    low =  1
    high = x
    feedback = ''
    while feedback != 'c':
        if low != high:
            guess = random.randint(low,high)
        else:
            guess = low
            feedback = input(f'is {guess} too high(H), too low(L), or correct (c) ')
            if feedback == 'h':
                high = guess - 1
            elif feedback == 'l':
                low = guess + 1
                print(f'yes! The computer guessed your number, {guess}, correctly')

import random
def play():
    user = input("what's your choise k' for knife, 'y' for yam, 'j' for jackfruit")
    computer = random.choice(['k','y','j'])
    if user == computer:
        return 'oops! tied'
    # k>y,y>j, j>k
    if is_win(user, computer ):
        return 'you win'
    # k>y,y>j, j>k

    else:
        return 'you loose'
def is_win(player, opponent):
    # return true if the user wins
    # k>y,y>j, j>k
    if (player =='k' and opponent =='y') or  (player=='y'and opponent=='j')\
              or( player=='j' and opponent=='k'):
        return 'true'



print(play())
