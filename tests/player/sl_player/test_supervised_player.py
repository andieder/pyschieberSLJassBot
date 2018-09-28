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
from sljassbot.player.sl_player.one_three_three_eight.sl_player import SLPlayer1338
from sljassbot.player.sl_player.human_convolution_one_three_three_eight.sl_player import SLPlayer1338hc


@pytest.fixture(scope='module')
def rl_models_directory():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return dir_path + '/rl_models/'


@pytest.fixture(scope='module')
def sl_models_directory():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return dir_path + '/sl_models/'


@pytest.fixture(scope='module')
def sl37_trumpf_model_path(sl_models_directory):
    return sl_models_directory + 'sl37_trumpf_model.h5'


@pytest.fixture(scope='module')
def sl186_model_path(sl_models_directory):
    return sl_models_directory + 'sl186_game_model.h5'


@pytest.fixture(scope='module')
def sl222_model_path(sl_models_directory):
    return sl_models_directory + 'sl222_game_model.h5'


@pytest.fixture(scope='module')
def sl3lp222_model_path(sl_models_directory):
    return sl_models_directory + 'sl3LP222_game_model.h5'


@pytest.fixture(scope='module')
def sl1338_big_nn_model_path(sl_models_directory):
    return sl_models_directory + 'sl1338_BigNN_game_model.h5'


@pytest.fixture(scope='module')
def sl1338_hc_model_path(sl_models_directory):
    return sl_models_directory + 'sl1338_human_convolution_game_model.h5'


@pytest.mark.statistical
def test_sl186_vs_random(sl186_model_path):
    players = [SLPlayer186(name='SLPlayer1', game_model_path=sl186_model_path), RandomPlayer(name='Track'),
               SLPlayer186(name='SLPlayer2', game_model_path=sl186_model_path), RandomPlayer(name='Trick')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl222_vs_random(sl222_model_path):
    players = [SLPlayer222(name='SLPlayer1', game_model_path=sl222_model_path), RandomPlayer(name='Track'),
               SLPlayer222(name='SLPlayer2', game_model_path=sl222_model_path), RandomPlayer(name='Trick')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl_3lp_222_vs_random(sl3lp222_model_path):
    players = [SLPlayer3LP222(name='SLPlayer1', game_model_path=sl3lp222_model_path), RandomPlayer(name='Track'),
               SLPlayer3LP222(name='SLPlayer2', game_model_path=sl3lp222_model_path), RandomPlayer(name='Trick')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl1338_vs_random(sl1338_big_nn_model_path):
    players = [SLPlayer1338(name='SLPlayer1', game_model_path=sl1338_big_nn_model_path), RandomPlayer(name='Track'),
               RandomPlayer(name='Tick'), RandomPlayer(name='Trick')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl1338hc_vs_random(sl1338_hc_model_path):
    players = [SLPlayer1338hc(name='SLPlayer1', game_model_path=sl1338_hc_model_path), RandomPlayer(name='Track'),
               SLPlayer1338hc(name='SLPlayer2', game_model_path=sl1338_hc_model_path), RandomPlayer(name='Trick')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl186_vs_random_with_trumpfnetwork(sl186_model_path, sl37_trumpf_model_path):
    players = [SLPlayer186(name='SLPlayer1', game_model_path=sl186_model_path, trumpf_model_path=sl37_trumpf_model_path),
               RandomPlayer(name='Track'),
               SLPlayer186(name='SLPlayer2', game_model_path=sl186_model_path, trumpf_model_path=sl37_trumpf_model_path),
               RandomPlayer(name='Trick')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl222_vs_random_with_trumpfnetwork(sl222_model_path, sl37_trumpf_model_path):
    players = [SLPlayer222(name='SLPlayer1', game_model_path=sl222_model_path, trumpf_model_path=sl37_trumpf_model_path),
               RandomPlayer(name='Track'),
               SLPlayer222(name='SLPlayer2', game_model_path=sl222_model_path, trumpf_model_path=sl37_trumpf_model_path),
               RandomPlayer(name='Trick')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl_3lp_222_vs_random_with_trumpfnetwork(sl3lp222_model_path, sl37_trumpf_model_path):
    players = [SLPlayer3LP222(name='SLPlayer1', game_model_path=sl3lp222_model_path, trumpf_model_path=sl37_trumpf_model_path),
               RandomPlayer(name='Track'),
               SLPlayer3LP222(name='SLPlayer2', game_model_path=sl3lp222_model_path, trumpf_model_path=sl37_trumpf_model_path),
               RandomPlayer(name='Trick')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl1338_vs_random_with_trumpfnetwork(sl1338_big_nn_model_path, sl37_trumpf_model_path):
    players = [SLPlayer1338(name='SLPlayer1', game_model_path=sl1338_big_nn_model_path, trumpf_model_path=sl37_trumpf_model_path),
               RandomPlayer(name='Tick'), RandomPlayer(name='Trick'), RandomPlayer(name='Track')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl1338hc_vs_random_with_trumpfnetwork(sl1338_hc_model_path, sl37_trumpf_model_path):
    players = [SLPlayer1338hc(name='SLPlayer1', game_model_path=sl1338_hc_model_path, trumpf_model_path=sl37_trumpf_model_path),
               RandomPlayer(name='Tick'),
               SLPlayer1338hc(name='SLPlayer2', game_model_path=sl1338_hc_model_path, trumpf_model_path=sl37_trumpf_model_path),
               RandomPlayer(name='Track')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl186_vs_greedy(sl186_model_path):
    players = [SLPlayer186(name='SLPlayer1', game_model_path=sl186_model_path), GreedyPlayer(name='Greedy1'),
               SLPlayer186(name='SLPlayer2', game_model_path=sl186_model_path), GreedyPlayer(name='Greedy2')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl222_vs_greedy(sl222_model_path):
    players = [SLPlayer222(name='SLPlayer1', game_model_path=sl222_model_path), GreedyPlayer(name='Greedy1'),
               SLPlayer222(name='SLPlayer2', game_model_path=sl222_model_path), GreedyPlayer(name='Greedy2')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl_3lp_222_vs_greedy(sl3lp222_model_path):
    players = [SLPlayer3LP222(name='SLPlayer1', game_model_path=sl3lp222_model_path), GreedyPlayer(name='Greedy1'),
               SLPlayer3LP222(name='SLPlayer2', game_model_path=sl3lp222_model_path), GreedyPlayer(name='Greedy2')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl1338_vs_greedy(sl1338_big_nn_model_path):
    players = [SLPlayer1338(name='SLPlayer1', game_model_path=sl1338_big_nn_model_path),
               GreedyPlayer(name='Greedy1'), GreedyPlayer(name='Greedy2'), GreedyPlayer(name='Greedy3')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl1338hc_vs_greedy(sl1338_hc_model_path):
    players = [SLPlayer1338hc(name='SLPlayer1', game_model_path=sl1338_hc_model_path),
               GreedyPlayer(name='Greedy1'),
               SLPlayer1338hc(name='SLPlayer1', game_model_path=sl1338_hc_model_path),
               GreedyPlayer(name='Greedy2')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl186_vs_challenge(sl186_model_path):
    players = [SLPlayer186(name='SLPlayer1', game_model_path=sl186_model_path), ChallengePlayer(name='ChallengePlayer1'),
               SLPlayer186(name='SLPlayer2', game_model_path=sl186_model_path), ChallengePlayer(name='ChallengePlayer2')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl222_vs_challenge(sl222_model_path):
    players = [SLPlayer222(name='SLPlayer1', game_model_path=sl222_model_path), ChallengePlayer(name='ChallengePlayer1'),
               SLPlayer222(name='SLPlayer2', game_model_path=sl222_model_path), ChallengePlayer(name='ChallengePlayer2')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl_3lp_222_vs_challenge(sl3lp222_model_path):
    players = [SLPlayer3LP222(name='SLPlayer1', game_model_path=sl3lp222_model_path), ChallengePlayer(name='ChallengePlayer1'),
               SLPlayer3LP222(name='SLPlayer2', game_model_path=sl3lp222_model_path), ChallengePlayer(name='ChallengePlayer2')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl1338_vs_challenge(sl1338_big_nn_model_path):
    players = [SLPlayer1338(name='SLPlayer1', game_model_path=sl1338_big_nn_model_path),
               ChallengePlayer(name='ChallengePlayer1'),
               ChallengePlayer(name='ChallengePlayer2'), ChallengePlayer(name='ChallengePlayer3')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl1338hc_vs_challenge(sl1338_hc_model_path):
    players = [SLPlayer1338hc(name='SLPlayer1', game_model_path=sl1338_hc_model_path),
               ChallengePlayer(name='ChallengePlayer1'),
               SLPlayer1338hc(name='SLPlayer2', game_model_path=sl1338_hc_model_path),
               ChallengePlayer(name='ChallengePlayer2')]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl186_vs_rl(sl186_model_path, rl_models_directory):
    rl_model_path = rl_models_directory + 'rl1_model.h5'
    players = [SLPlayer186(name='SLPlayer1', game_model_path=sl186_model_path),
               RLPlayer(name='RLPlayer1', model_path=rl_model_path),
               SLPlayer186(name='SLPlayer2', game_model_path=sl186_model_path),
               RLPlayer(name='RLPlayer2', model_path=rl_model_path)]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl222_vs_rl(sl222_model_path, rl_models_directory):
    rl_model_path = rl_models_directory + 'rl1_model.h5'
    players = [SLPlayer222(name='SLPlayer1', game_model_path=sl222_model_path),
               RLPlayer(name='RLPlayer1', model_path=rl_model_path),
               SLPlayer222(name='SLPlayer2', game_model_path=sl222_model_path),
               RLPlayer(name='RLPlayer2', model_path=rl_model_path)]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl_3lp_222_vs_rl(sl3lp222_model_path, rl_models_directory):
    rl_model_path = rl_models_directory + 'rl1_model.h5'
    players = [SLPlayer3LP222(name='SLPlayer1', game_model_path=sl3lp222_model_path),
               RLPlayer(name='RLPlayer1', model_path=rl_model_path),
               SLPlayer3LP222(name='SLPlayer2', game_model_path=sl3lp222_model_path),
               RLPlayer(name='RLPlayer2', model_path=rl_model_path)]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl1338_vs_rl(sl1338_big_nn_model_path, rl_models_directory):
    rl_model_path = rl_models_directory + 'rl1_model.h5'
    players = [SLPlayer1338(name='SLPlayer1', game_model_path=sl1338_big_nn_model_path),
               RLPlayer(name='RLPlayer1', model_path=rl_model_path),
               RLPlayer(name='RLPlayer2', model_path=rl_model_path),
               RLPlayer(name='RLPlayer3', model_path=rl_model_path)]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl1338hc_vs_rl(sl1338_hc_model_path, rl_models_directory):
    rl_model_path = rl_models_directory + 'rl1_model.h5'
    players = [SLPlayer1338hc(name='SLPlayer1', game_model_path=sl1338_hc_model_path),
               RLPlayer(name='RLPlayer1', model_path=rl_model_path),
               SLPlayer1338hc(name='SLPlayer2', game_model_path=sl1338_hc_model_path),
               RLPlayer(name='RLPlayer2', model_path=rl_model_path)]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl186_vs_sl222(sl186_model_path, sl222_model_path):
    players = [SLPlayer186(name='SLPlayer186_1', game_model_path=sl186_model_path),
               SLPlayer222(name='SLPlayer222_1', game_model_path=sl222_model_path),
               SLPlayer186(name='SLPlayer186_2', game_model_path=sl186_model_path),
               SLPlayer222(name='SLPlayer222_2', game_model_path=sl222_model_path)]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl186_vs_sl_3lp_222(sl186_model_path, sl3lp222_model_path):
    players = [SLPlayer186(name='SLPlayer186_1', game_model_path=sl186_model_path),
               SLPlayer3LP222(name='SLPlayer3LP222_1', game_model_path=sl3lp222_model_path),
               SLPlayer186(name='SLPlayer186_2', game_model_path=sl186_model_path),
               SLPlayer3LP222(name='SLPlayer3LP222_2', game_model_path=sl3lp222_model_path)]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl186_vs_sl1338(sl186_model_path, sl1338_big_nn_model_path):
    players = [SLPlayer186(name='SLPlayer186_1', game_model_path=sl186_model_path),
               SLPlayer1338(name='SLPlayer1338_1', game_model_path=sl1338_big_nn_model_path),
               SLPlayer186(name='SLPlayer186_2', game_model_path=sl186_model_path),
               SLPlayer186(name='SLPlayer186_3', game_model_path=sl186_model_path)]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl186_vs_sl1338hc(sl186_model_path, sl1338_hc_model_path):
    players = [SLPlayer186(name='SLPlayer186_1', game_model_path=sl186_model_path),
               SLPlayer1338hc(name='SLPlayer1338_1', game_model_path=sl1338_hc_model_path),
               SLPlayer186(name='SLPlayer186_2', game_model_path=sl186_model_path),
               SLPlayer1338hc(name='SLPlayer1338_2', game_model_path=sl1338_hc_model_path)]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl222_vs_sl_3lp_222(sl222_model_path, sl3lp222_model_path):
    players = [SLPlayer222(name='SLPlayer222_1', game_model_path=sl222_model_path),
               SLPlayer3LP222(name='SLPlayer3LP222_1', game_model_path=sl3lp222_model_path),
               SLPlayer222(name='SLPlayer222_2', game_model_path=sl222_model_path),
               SLPlayer3LP222(name='SLPlayer3LP222_2', game_model_path=sl3lp222_model_path)]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl222_vs_sl1338(sl222_model_path, sl1338_big_nn_model_path):
    players = [SLPlayer222(name='SLPlayer222_1', game_model_path=sl222_model_path),
               SLPlayer1338(name='SLPlayer1338_1', game_model_path=sl1338_big_nn_model_path),
               SLPlayer222(name='SLPlayer222_2', game_model_path=sl222_model_path),
               SLPlayer222(name='SLPlayer222_3', game_model_path=sl222_model_path)]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl222_vs_sl1338hc(sl222_model_path, sl1338_hc_model_path):
    players = [SLPlayer222(name='SLPlayer222_1', game_model_path=sl222_model_path),
               SLPlayer1338hc(name='SLPlayer1338hc_1', game_model_path=sl1338_hc_model_path),
               SLPlayer222(name='SLPlayer222_2', game_model_path=sl222_model_path),
               SLPlayer1338hc(name='SLPlayer1338hc_2', game_model_path=sl1338_hc_model_path)]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl1338_vs_sl_3lp_222(sl1338_big_nn_model_path, sl3lp222_model_path):
    players = [SLPlayer1338(name='SLPlayer1338_1', game_model_path=sl1338_big_nn_model_path),
               SLPlayer3LP222(name='SLPlayer3LP222_1', game_model_path=sl3lp222_model_path),
               SLPlayer3LP222(name='SLPlayer3LP222_2', game_model_path=sl3lp222_model_path),
               SLPlayer3LP222(name='SLPlayer3LP222_3', game_model_path=sl3lp222_model_path)]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl1338hc_vs_sl_3lp_222(sl1338_hc_model_path, sl3lp222_model_path):
    players = [SLPlayer1338hc(name='SLPlayer1338hc_1', game_model_path=sl1338_hc_model_path),
               SLPlayer3LP222(name='SLPlayer3LP222_1', game_model_path=sl3lp222_model_path),
               SLPlayer1338hc(name='SLPlayer1338hc_2', game_model_path=sl1338_hc_model_path),
               SLPlayer3LP222(name='SLPlayer3LP222_2', game_model_path=sl3lp222_model_path)]
    get_function_name()
    run_game(players)


@pytest.mark.statistical
def test_sl1338_vs_sl1338hc(sl1338_big_nn_model_path, sl1338_hc_model_path):
    players = [SLPlayer1338(name='SLPlayer1338_1', game_model_path=sl1338_big_nn_model_path),
               SLPlayer1338hc(name='SLPlayer1338hc_1', game_model_path=sl1338_hc_model_path),
               SLPlayer1338hc(name='SLPlayer1338hc_2', game_model_path=sl1338_hc_model_path),
               SLPlayer1338hc(name='SLPlayer1338hc_3', game_model_path=sl1338_hc_model_path)]
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
