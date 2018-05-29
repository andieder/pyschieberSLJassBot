import os
import inspect
from timeit import default_timer as timer

import pytest
from pyschieber.player.greedy_player.greedy_player import GreedyPlayer
from pyschieber.player.challenge_player.challenge_player import ChallengePlayer

from pyschieber.player.random_player import RandomPlayer
from pyschieber.tournament import Tournament

from sljassbot.player.rl_player.rl_player import RLPlayer


@pytest.fixture(scope='module')
def weights_directory():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return dir_path + '/weights/'


@pytest.mark.statistical  #
def test_random_vs_random():
    players = [RandomPlayer(name='Tick'), RandomPlayer(name='Track'),
               RandomPlayer(name='Dagobert'), RandomPlayer(name='Trick')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical  #
def test_greedy_vs_greedy():
    players = [GreedyPlayer(name='Greedy1'), GreedyPlayer(name='Greedy2'),
               GreedyPlayer(name='Greedy3'), GreedyPlayer(name='Greedy4')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical  #
def test_greedy_vs_random():
    players = [GreedyPlayer(name='Greedy1'), RandomPlayer(name='Random1'),
               GreedyPlayer(name='Greedy2'), RandomPlayer(name='Random2')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_rl_vs_random(weights_directory):
    model_path = weights_directory + 'rl1_model.h5'
    players = [RLPlayer(name='RLPlayer1', model_path=model_path), RandomPlayer(name='Track'),
               RLPlayer(name='RLPlayer2', model_path=model_path), RandomPlayer(name='Trick')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_rl_vs_greedy(weights_directory):
    model_path = weights_directory + 'rl1_model.h5'
    players = [RLPlayer(name='RLPlayer1', model_path=model_path), GreedyPlayer(name='Greedy1'),
               RLPlayer(name='RLPlayer2', model_path=model_path), GreedyPlayer(name='Greedy2')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical  #
def test_greedy_vs_challenge():
    players = [GreedyPlayer(name='GreedyPlayer1'), ChallengePlayer(name='ChallengePlayer1'),
               GreedyPlayer(name='GreedyPlayer2'), ChallengePlayer(name='ChallengePlayer2')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_rl_vs_challenge(weights_directory):
    model_path = weights_directory + 'rl1_model.h5'
    players = [RLPlayer(name='RLPlayer1', model_path=model_path), ChallengePlayer(name='ChallengePlayer1'),
               RLPlayer(name='RLPlayer2', model_path=model_path), ChallengePlayer(name='ChallengePlayer2')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_rl_greedy_vs_rl_challenge(weights_directory):
    model_path = weights_directory + 'rl1_model_challenge.h5'
    model_path_greedy = weights_directory + 'rl1_model.h5'
    players = [RLPlayer(name='RLGreedyPlayer1', model_path=model_path_greedy),
               RLPlayer(name='RLChallengePlayer1', model_path=model_path),
               RLPlayer(name='RLGreedyPlayer2', model_path=model_path_greedy),
               RLPlayer(name='RLChallengePlayer2', model_path=model_path)]
    get_function_name()
    run_game(players)


@pytest.mark.statistical  #
def test_greedy_and_challenge_vs_2_challenge():
    players = [GreedyPlayer(name='Gready1'), ChallengePlayer(name='ChallengePlayer1'),
               ChallengePlayer(name='ChallengePlayer2'), ChallengePlayer(name='ChallengePlayer3')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical  #
def test_random_and_challenge_vs_2_challenge():
    players = [RandomPlayer(name='Random1'), ChallengePlayer(name='ChallengePlayer1'),
               ChallengePlayer(name='ChallengePlayer2'), ChallengePlayer(name='ChallengePlayer3')]
    get_function_name()
    run_game(players)


def run_game(players, point_limit=1000, number_of_tournaments=1000):
    tournament = Tournament(point_limit=point_limit)
    [tournament.register_player(player=player) for player in players]

    team_1_won = 0
    team_2_won = 0

    start = timer()

    for _ in range(number_of_tournaments):
        tournament.play()
        if tournament.teams[0].won(point_limit=point_limit):
            team_1_won += 1
        else:
            team_2_won += 1

    end = timer()
    print("\nTo run {0} tournaments it took {1:.2f} seconds.".format(number_of_tournaments, end - start))

    difference = abs(team_1_won - team_2_won)
    print("Difference: ", difference)
    print("Team 1 ({0}, {1}): {2}".format(players[0].name, players[2].name, team_1_won))
    print("Team 2 ({0}, {1}): {2}".format(players[1].name, players[3].name, team_2_won))
    assert True


def get_function_name():
    print("\n \n{}".format(inspect.stack()[1][3]))