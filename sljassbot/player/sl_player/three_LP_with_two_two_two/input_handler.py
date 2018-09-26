import numpy as np
from matplotlib import pyplot as plt

from pyschieber.card import from_string_to_card
from pyschieber.deck import Deck
from pyschieber.trumpf import Trumpf

'''
|                               played cards
|    hand cards   |      table       |    player 0     |    player 1      |  player 2      |  player 3      |trumpf|
|-----------------|------------------|-----------------|------------------|----------------|----------------|------|

6 * 36 + 6 = 222

Split the network in three learning parts - Three networks.
1. network for the first three cards (1.-3. cards) -> first_game_network
2. network for the second three cards (4.-6. cards) -> second_game_network
3. network for the last three cards (7.-9. cards) -> third_game_network

|   first_network   |   second_network  |   third_network   |
|--------...--------|--------...--------|--------...--------|

3 * 222 = 666

Every network has the same input and the same output layer, but the three game networks are independent of each other.

The output layer of each network has 36 neurons or 36 possible cards.
'''


class InputHandler:
    deck = Deck()
    trumpfs = [trumpf.name for trumpf in list(Trumpf)]

    nr_cards = 36
    nr_player = 4
    nr_trumpf_modes = 6

    pos_hand_cards = 0
    pos_table = 1 * nr_cards
    pos_player_played_card = [2 * nr_cards, 3 * nr_cards, 4 * nr_cards, 5 * nr_cards]
    pos_trumpf = 6 * nr_cards

    network_size = 6 * nr_cards + nr_trumpf_modes
    input_size = 3 * network_size
    output_size = 3 * nr_cards

    def __init__(self):
        self.state = None
        self.network_preparation = None
        self.reset()

    def reset(self):
        self.state = np.zeros(self.input_size, dtype='float32')
        self.network_preparation = np.zeros(self.network_size, dtype='float32')

    def update_state_stich(self, game_state, cards, player_id):
        table_list = analyse_table(game_state['table'])
        self.set_player(table_list, player_id)

    def update_state_choose_card(self, game_state, cards, player_id):
        table_list = analyse_table(game_state['table'])
        self.set_hand(cards)
        self.set_table(table_list)
        self.set_player(table_list, player_id)
        self.set_trumpf(trumpf=game_state['trumpf'])
        self.state[self.pos_hand_cards:self.input_size - 1] = 0.

        amount_hand_cards = len(cards)
        if 10 > amount_hand_cards > 6:
            self.set_state(0)
        elif 0 < amount_hand_cards < 4:
            self.set_state(2)
        elif 3 < amount_hand_cards < 7:
            self.set_state(1)

        self.network_preparation[self.pos_hand_cards:self.network_size - 1] = 0.

    def set_player(self, table, current_player_id):
        self.network_preparation[self.pos_player_played_card[0]:self.pos_player_played_card[len(
            self.pos_player_played_card) - 1] + self.nr_cards] = 0.
        player_id = 0
        for id, card in table:
            if id != current_player_id:
                self.network_preparation[self.pos_player_played_card[player_id] + card_to_index(card)] = 1.
                player_id += 1

    def set_table(self, table):
        self.network_preparation[self.pos_table:self.pos_table + self.nr_cards] = 0.
        for _, card in table:
            self.network_preparation[self.pos_table + card_to_index(card)] = 1.

    def set_hand(self, cards):
        self.network_preparation[self.pos_hand_cards:self.pos_hand_cards + self.nr_cards] = 0.
        for card in cards:
            self.network_preparation[self.pos_hand_cards + card_to_index(card)] = 1.

    def set_trumpf(self, trumpf):
        self.network_preparation[self.pos_trumpf + self.trumpfs.index(trumpf)] = 1.

    def set_state(self, network_number):
        network_preparation_list = self.network_preparation[0:self.network_size].tolist()
        for index in range(0, len(network_preparation_list)):
            self.state[network_number * self.network_size + index] = network_preparation_list[index]


def card_to_index(card):
    return InputHandler.deck.cards.index(card)


def index_to_card(index):
    return InputHandler.deck.cards[index]


def analyse_table(table):
    table_list = []
    for played_card in table:
        player_id = played_card['player_id']
        card = from_string_to_card(played_card['card'])
        table_list.append((player_id, card))
    return table_list


def print_state(input_state, player_id):
    y = np.reshape(input_state, (1, InputHandler.input_size))
    plt.imshow(y, cmap='gray')
    title_obj = plt.title('Player: {}'.format(player_id))
    plt.setp(title_obj, color='black')
    plt.show()


def count_played_cards(input_state, player_id):
    return np.count_nonzero(input_state[
                            InputHandler.pos_player_played_card[player_id]:InputHandler.pos_player_played_card[
                                                                               player_id] + InputHandler.nr_cards])
