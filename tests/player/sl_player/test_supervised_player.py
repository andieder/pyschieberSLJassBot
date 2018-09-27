import os
import inspect
from timeit import default_timer as timer

import pytest
from pyschieber.player.random_player import RandomPlayer
from pyschieber.player.greedy_player.greedy_player import GreedyPlayer
from pyschieber.player.challenge_player.challenge_player import ChallengePlayer

from sljassbot.player.rl_player.rl_player import RLPlayer

from pyschieber.tournament import Tournament

from sljassbot.player.sl_player.one_eight_six.sl_player import SLPlayer186
from sljassbot.player.sl_player.two_two_two.sl_player import SLPlayer222
from sljassbot.player.sl_player.three_LP_with_two_two_two.sl_player import SLPlayer3LP222
# from sljassbot.player.sl_player.one_three_three_eight.sl_player import SLPlayer1338


@pytest.fixture(scope='module')
def rl_models_directory():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return dir_path + '/rl_models/'


@pytest.fixture(scope='module')
def sl_models_directory():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return dir_path + '/sl_models/'


@pytest.mark.statistical
def test_sl186_vs_random(sl_models_directory):
    model_path = sl_models_directory + 'sl186_game_model.h5'
    players = [SLPlayer186(name='SLPlayer1', game_model_path=model_path), RandomPlayer(name='Track'),
               SLPlayer186(name='SLPlayer2', game_model_path=model_path), RandomPlayer(name='Trick')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl222_vs_random(sl_models_directory):
    model_path = sl_models_directory + 'sl222_game_model.h5'
    players = [SLPlayer222(name='SLPlayer1', game_model_path=model_path), RandomPlayer(name='Track'),
               SLPlayer222(name='SLPlayer2', game_model_path=model_path), RandomPlayer(name='Trick')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl_3lp_222_vs_random(sl_models_directory):
    model_path = sl_models_directory + 'sl3LP222_game_model.h5'
    players = [SLPlayer3LP222(name='SLPlayer1', game_model_path=model_path), RandomPlayer(name='Track'),
               SLPlayer3LP222(name='SLPlayer2', game_model_path=model_path), RandomPlayer(name='Trick')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl186_vs_random_with_trumpfnetwork(sl_models_directory):
    model_path = sl_models_directory + '/sl186_game_model.h5'
    trumpf_model_path = sl_models_directory + '/sl37_trumpf_model.h5'
    players = [SLPlayer186(name='SLPlayer1', game_model_path=model_path, trumpf_model_path=trumpf_model_path),
               RandomPlayer(name='Track'),
               SLPlayer186(name='SLPlayer2', game_model_path=model_path, trumpf_model_path=trumpf_model_path),
               RandomPlayer(name='Trick')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl222_vs_random_with_trumpfnetwork(sl_models_directory):
    model_path = sl_models_directory + '/sl222_game_model.h5'
    trumpf_model_path = sl_models_directory + '/sl37_trumpf_model.h5'
    players = [SLPlayer222(name='SLPlayer1', game_model_path=model_path, trumpf_model_path=trumpf_model_path),
               RandomPlayer(name='Track'),
               SLPlayer222(name='SLPlayer2', game_model_path=model_path, trumpf_model_path=trumpf_model_path),
               RandomPlayer(name='Trick')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl_3lp_222_vs_random_with_trumpfnetwork(sl_models_directory):
    model_path = sl_models_directory + '/sl3LP222_game_model.h5'
    trumpf_model_path = sl_models_directory + '/sl37_trumpf_model.h5'
    players = [SLPlayer3LP222(name='SLPlayer1', game_model_path=model_path, trumpf_model_path=trumpf_model_path),
               RandomPlayer(name='Track'),
               SLPlayer3LP222(name='SLPlayer2', game_model_path=model_path, trumpf_model_path=trumpf_model_path),
               RandomPlayer(name='Trick')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl186_vs_greedy(sl_models_directory):
    model_path = sl_models_directory + 'sl186_game_model.h5'
    players = [SLPlayer186(name='SLPlayer1', game_model_path=model_path), GreedyPlayer(name='Greedy1'),
               SLPlayer186(name='SLPlayer2', game_model_path=model_path), GreedyPlayer(name='Greedy2')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl222_vs_greedy(sl_models_directory):
    model_path = sl_models_directory + 'sl222_game_model.h5'
    players = [SLPlayer222(name='SLPlayer1', game_model_path=model_path), GreedyPlayer(name='Greedy1'),
               SLPlayer222(name='SLPlayer2', game_model_path=model_path), GreedyPlayer(name='Greedy2')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl_3lp_222_vs_greedy(sl_models_directory):
    model_path = sl_models_directory + 'sl3LP222_game_model.h5'
    players = [SLPlayer3LP222(name='SLPlayer1', game_model_path=model_path), GreedyPlayer(name='Greedy1'),
               SLPlayer3LP222(name='SLPlayer2', game_model_path=model_path), GreedyPlayer(name='Greedy2')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl186_vs_challenge(sl_models_directory):
    model_path = sl_models_directory + 'sl186_game_model.h5'
    players = [SLPlayer186(name='SLPlayer1', game_model_path=model_path), ChallengePlayer(name='ChallengePlayer1'),
               SLPlayer186(name='SLPlayer2', game_model_path=model_path), ChallengePlayer(name='ChallengePlayer2')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl222_vs_challenge(sl_models_directory):
    model_path = sl_models_directory + 'sl222_game_model.h5'
    players = [SLPlayer222(name='SLPlayer1', game_model_path=model_path), ChallengePlayer(name='ChallengePlayer1'),
               SLPlayer222(name='SLPlayer2', game_model_path=model_path), ChallengePlayer(name='ChallengePlayer2')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl_3lp_222_vs_challenge(sl_models_directory):
    model_path = sl_models_directory + 'sl3LP222_game_model.h5'
    players = [SLPlayer3LP222(name='SLPlayer1', game_model_path=model_path), ChallengePlayer(name='ChallengePlayer1'),
               SLPlayer3LP222(name='SLPlayer2', game_model_path=model_path), ChallengePlayer(name='ChallengePlayer2')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl186_vs_rl(sl_models_directory, rl_models_directory):
    sl_model_path = sl_models_directory + 'sl186_game_model.h5'
    rl_model_path = rl_models_directory + 'rl1_model.h5'
    players = [SLPlayer186(name='SLPlayer1', game_model_path=sl_model_path),
               RLPlayer(name='RLPlayer1', model_path=rl_model_path),
               SLPlayer186(name='SLPlayer2', game_model_path=sl_model_path),
               RLPlayer(name='RLPlayer2', model_path=rl_model_path)]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl222_vs_rl(sl_models_directory, rl_models_directory):
    sl_model_path = sl_models_directory + 'sl222_game_model.h5'
    rl_model_path = rl_models_directory + 'rl1_model.h5'
    players = [SLPlayer222(name='SLPlayer1', game_model_path=sl_model_path),
               RLPlayer(name='RLPlayer1', model_path=rl_model_path),
               SLPlayer222(name='SLPlayer2', game_model_path=sl_model_path),
               RLPlayer(name='RLPlayer2', model_path=rl_model_path)]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl_3lp_222_vs_rl(sl_models_directory, rl_models_directory):
    sl_model_path = sl_models_directory + 'sl3LP222_game_model.h5'
    rl_model_path = rl_models_directory + 'rl1_model.h5'
    players = [SLPlayer3LP222(name='SLPlayer1', game_model_path=sl_model_path),
               RLPlayer(name='RLPlayer1', model_path=rl_model_path),
               SLPlayer3LP222(name='SLPlayer2', game_model_path=sl_model_path),
               RLPlayer(name='RLPlayer2', model_path=rl_model_path)]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl186_vs_sl222(sl_models_directory):
    sl186_model_path = sl_models_directory + 'sl186_game_model.h5'
    sl222_model_path = sl_models_directory + 'sl222_game_model.h5'
    players = [SLPlayer186(name='SLPlayer186_1', game_model_path=sl186_model_path),
               SLPlayer222(name='SLPlayer222_1', game_model_path=sl222_model_path),
               SLPlayer186(name='SLPlayer186_2', game_model_path=sl186_model_path),
               SLPlayer222(name='SLPlayer222_2', game_model_path=sl222_model_path)]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl186_vs_sl_3lp_222(sl_models_directory):
    sl186_model_path = sl_models_directory + 'sl186_game_model.h5'
    sl_3lp_222_model_path = sl_models_directory + 'sl3LP222_game_model.h5'
    players = [SLPlayer186(name='SLPlayer186_1', game_model_path=sl186_model_path),
               SLPlayer3LP222(name='SLPlayer3LP222_1', game_model_path=sl_3lp_222_model_path),
               SLPlayer186(name='SLPlayer186_2', game_model_path=sl186_model_path),
               SLPlayer3LP222(name='SLPlayer3LP222_2', game_model_path=sl_3lp_222_model_path)]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl222_vs_sl_3lp_222(sl_models_directory):
    sl222_model_path = sl_models_directory + 'sl222_game_model.h5'
    sl_3lp_222_model_path = sl_models_directory + 'sl3LP222_game_model.h5'
    players = [SLPlayer222(name='SLPlayer222_1', game_model_path=sl222_model_path),
               SLPlayer3LP222(name='SLPlayer3LP222_1', game_model_path=sl_3lp_222_model_path),
               SLPlayer222(name='SLPlayer222_2', game_model_path=sl222_model_path),
               SLPlayer3LP222(name='SLPlayer3LP222_2', game_model_path=sl_3lp_222_model_path)]
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
