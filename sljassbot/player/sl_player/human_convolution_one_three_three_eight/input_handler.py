import numpy as np
from matplotlib import pyplot as plt

from pyschieber.card import from_string_to_card
from pyschieber.deck import Deck
from pyschieber.trumpf import Trumpf

'''
|  36 hand cards  |   table cards (36 cards per player)   |   history (36 cards per stich per player)    | trumpf |
|-----------------|---------------------------------------|----------------------------------------------|--------|


|          36 table cards per each player           |
|---------------------------------------------------|
|| player 1  |  player 2  |  player 3  |  player 4 ||
|------------|------------|------------|------------|
-->> This is the current stich, because that are the most important cards for the current desicion
-->> That is the reason too, why we make the history just to 8 stichs 


|          36 cards per stich per player (history of the stichs from the current game without the last stich)         | 
|---------------------------------------------------------------------------------------------------------------------|
|           stich 1         ||           stich 2         ||           stich 3         |...|           stich 8         |
|---------------------------||---------------------------||---------------------------|...|---------------------------|
|   p1 |   p2 |   p3 |   p4 ||   p1 |   p2 |   p3 |   p4 ||   p1 |   p2 |   p3 |   p4 |...|   p1 |   p2 |   p3 |   p4 |


36 + 4 * 36 + 8 * 4 * 36 + 6 = 1'338
This network has a human convolution in the first 2 - 3 hidden layers to a common size of 350 neurons.
Behind this layer the network grow and shink down in five hidden layers. 
'''


class InputHandler:
    deck = Deck()
    trumpfs = [trumpf.name for trumpf in list(Trumpf)]

    nr_cards = 36
    nr_trumpf_modes = 6
    amount_players = 4
    amount_card_sets = 37

    pos_hand_cards = 0
    pos_player_on_table = [1 * nr_cards, 2 * nr_cards, 3 * nr_cards, 4 * nr_cards]
    # pos_players_per_stich is the history
    pos_trumpf = amount_card_sets * nr_cards

    input_size = amount_card_sets * nr_cards + nr_trumpf_modes  # 1338 input neurons
    output_size = nr_cards

    def __init__(self):
        self.state = None
        self.reset()
        self.pos_players_per_stich = self.generate_player_per_stich(self.amount_players + 1, self.amount_card_sets)

    def reset(self):
        self.state = np.zeros(self.input_size, dtype='float32')

    def update_state_stich(self, game_state, cards, player_id):
        table_list = analyse_table(game_state['table'])
        self.set_history(table_list, cards)

    def update_state_choose_card(self, game_state, cards, player_id):
        table_list = analyse_table(game_state['table'])
        self.set_hand(cards)
        self.set_table(table_list)
        self.set_history(table_list, cards)
        self.set_trumpf(trumpf=game_state['trumpf'])

    def set_history(self, table, cards):
        max_history_stich = 8
        amount_hand_cards = len(cards)
        stich_number = max_history_stich - amount_hand_cards
        if stich_number < max_history_stich:
            for player_id, card in table:
                self.state[self.pos_players_per_stich[stich_number][player_id] + card_to_index(card)] = 1.

    def set_table(self, table):
        self.state[self.pos_player_on_table[0]:self.pos_player_on_table[3] + self.nr_cards] = 0
        for player_position, card in table:
            self.state[self.pos_player_on_table[player_position] + card_to_index(card)] = 1.

    def set_hand(self, cards):
        self.state[self.pos_hand_cards:self.pos_hand_cards + self.nr_cards] = 0.
        for card in cards:
            self.state[self.pos_hand_cards + card_to_index(card)] = 1.

    def set_trumpf(self, trumpf):
        self.state[self.pos_trumpf + self.trumpfs.index(trumpf)] = 1.

    def generate_player_per_stich(self, start_range, end_range):
        player_collector = []
        pos_players_in_stich = []

        for player in range(start_range, end_range):
            player_collector.append(player * self.nr_cards)
            if len(player_collector) == 4 or player == end_range:
                pos_players_in_stich.append(player_collector)
                player_collector = []

        return pos_players_in_stich


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
