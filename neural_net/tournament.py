from random import shuffle
from neural_net.play_game import play_game

def generate_tournament():
    teams = ['simple_small', 'simple_large', 'complex_small', 'complex_large']
    shuffle(teams)
    matches = [[teams[0], teams[1]],
            [teams[2], teams[3]],
            [teams[1], teams[2]],
            [teams[3], teams[0]],
            [teams[2], teams[0]],
            [teams[3], teams[1]]]

    return matches, teams

def display(matches):
     for i, match in enumerate(matches):
         print(f"Match {i+1}: {match[0]} (W) vs {match[1]} (B)")

def run_tournament(matches, teams):
    scores = {team: 0 for team in teams}
    for i, match in enumerate(matches):
        winner = play_game(match[0], match[1], i)
        if winner == -1:
            scores[match[0]] += 1
            scores[match[1]] += 1
        elif winner == 0:
            scores[match[0]] += 3
        else:
            scores[match[1]] += 3

    return scores 

if __name__ == '__main__':
    matches, teams = generate_tournament()
    display(matches)
    scores = run_tournament(matches, teams)
