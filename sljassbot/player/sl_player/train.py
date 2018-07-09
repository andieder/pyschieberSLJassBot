import argparse
import os

from pyschieber.player.greedy_player.greedy_player import GreedyPlayer
from pyschieber.tournament import Tournament
from pyschieber.player.challenge_player.challenge_player import ChallengePlayer
from sljassbot.player.sl_player.sl_player import SLPlayer


def run(log_dir, episodes, rounds):
    model_path = log_dir + '/rl1_model.h5'
    trumpf_model_path = '/trumpf_network_model_final__2018-06-18_112248.h5'
    game_model_path = '/game_network_model_18__2017-11-16_161619.h5'
    sl_player = SLPlayer(name='SL1', model_path=game_model_path, rounds=rounds)
    # sl_player = SLPlayer(name='SL1', model_path=model_path, rounds=rounds)
    # players = [sl_player, ChallengePlayer(name='Tick'), ChallengePlayer(name='Trick'), ChallengePlayer(name='Track')]
    players = [sl_player, GreedyPlayer(name='Tick'), GreedyPlayer(name='Trick'), GreedyPlayer(name='Track')]
    sum_won = 0
    for e in range(episodes):
        tournament = Tournament()
        [tournament.register_player(player) for player in players]
        tournament.play(rounds=rounds, use_counting_factor=False)
        # sl_player.replay()
        sum_won += sl_player.won[0]
        print_stats_winning(sl_player.won_stich, sl_player.won, e, sum_won)
        sl_player.reset_stats()
    sl_player.model.save(model_path)


def print_stats_winning(won_stich, won, epoch, sum_won):
    text = ('-' * 180) + '\n'
    epoch += 1
    for i, win in enumerate(won_stich):
        text += "Player {0}: {1} \n".format(i, win)
    text += "Team 1: {0}\n".format(won[0])
    text += "Team 2: {0}\n".format(won[1])
    text += "Remis: {0}\n".format(won[2])
    text += "Epoche: {0}\n".format(epoch)
    text += "Mean won (team1): {0}\n".format(sum_won / epoch)
    print(text)
    with open('log.txt', 'a') as f:
        print(text, file=f)


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description='SchieberJassBot', )
    parser.add_argument('-l', '--log_dir', dest='log_dir', help='Tensorboard log directory')
    parser.add_argument('-e', '--nr_episodes', dest='nr_episodes', help='Number of episodes to play', type=int)
    parser.add_argument('-r', '--rounds', dest='rounds', help='Game rounds', type=int)
    parser.set_defaults(log_dir=dir_path + '/models', nr_episodes=10, rounds=20)
    args = parser.parse_args()
    run(log_dir=args.log_dir, episodes=args.nr_episodes, rounds=args.rounds)
