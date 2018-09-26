import numpy as np
from matplotlib import pyplot as plt

from pyschieber.card import from_string_to_card
from pyschieber.deck import Deck
from pyschieber.trumpf import Trumpf

'''
|                               played cards
|    hand cards   |    player 1     |    player 2      |  player 3      |  card counter  |trumpf|
|-----------------|-----------------|------------------|----------------|----------------|------|

5 * 36 + 6= 186
'''


class InputHandler:
    deck = Deck()
    trumpfs = [trumpf.name for trumpf in list(Trumpf)]

    nr_cards = 36
    nr_player = 4
    nr_trumpf_modes = 6

    pos_player_played_card = [1 * nr_cards, 2 * nr_cards, 3 * nr_cards]
    pos_trumpf = 5 * nr_cards
    pos_hand_cards = 0
    pos_card_counter = 4 * nr_cards

    input_size = 5 * nr_cards + nr_trumpf_modes
    output_size = nr_cards

    def __init__(self):
        self.state = None
        self.reset()

    def reset(self):
        self.state = np.zeros(self.input_size, dtype='float32')

    def update_state_stich(self, game_state, cards, player_id):
        table_list = analyse_table(game_state['table'])
        self.set_player(table_list, player_id)

    def update_state_choose_card(self, game_state, cards, player_id):
        table_list = analyse_table(game_state['table'])
        self.set_hand(cards)
        self.set_table(table_list)
        self.set_player(table_list, player_id)
        self.set_trumpf(trumpf=game_state['trumpf'])

    def set_player(self, table, current_player_id):
        self.state[self.pos_player_played_card[0]:self.pos_player_played_card[len(self.pos_player_played_card)-1] + self.nr_cards] = 0.
        player_id = 0
        for id, card in table:
            if id != current_player_id:
                self.state[self.pos_player_played_card[player_id] + card_to_index(card)] = 1.
                player_id += 1

    def set_table(self, table):
        for _, card in table:
            self.state[self.pos_card_counter + card_to_index(card)] = 1.

    def set_hand(self, cards):
        self.state[self.pos_hand_cards:self.pos_hand_cards + self.nr_cards] = 0.
        for card in cards:
            self.state[self.pos_hand_cards + card_to_index(card)] = 1.

    def set_trumpf(self, trumpf):
        self.state[self.pos_trumpf + self.trumpfs.index(trumpf)] = 1.


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
