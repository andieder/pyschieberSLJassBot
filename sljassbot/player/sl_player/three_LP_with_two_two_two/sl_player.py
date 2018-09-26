import logging
import random
import operator
from collections import deque

import numpy as np

from pyschieber.player.base_player import BasePlayer
from pyschieber.trumpf import Trumpf
from pyschieber.card import from_string_to_card
from pyschieber.player.greedy_player.trumpf_decision import choose_trumpf
from sljassbot.player.sl_player.three_LP_with_two_two_two.input_handler import InputHandler, card_to_index, index_to_card
from sljassbot.player.sl_player.three_LP_with_two_two_two.model import build_model

logger = logging.getLogger(__name__)


class SLPlayer3LP222(BasePlayer):
    max_points = 55
    min_points = -55

    def __init__(self, name, game_model_path, trumpf_model_path=None, rounds=1):
        super().__init__(name=name)
        self.input_handler_game_network = InputHandler()
        self.input_handler_trumpf_network = 36 + 1
        self.n_samples = 20 * rounds
        self.memories = deque([], maxlen=2 * self.n_samples)
        self.game_model = build_model(model_path=game_model_path)
        self.trumpf_model = build_model(model_path=trumpf_model_path)
        self.epsilon = 0.95  # exploration rate
        self.won = 0
        self.current_memory = dict(used=False)
        self.previous_memory = dict(used=False)
        self.team_points = [0, 0]
        self.won_stich = [0, 0, 0, 0]
        self.won = [0, 0, 0]

    def remember(self, state, action, reward, next_state, done, hand_cards, table_cards, trumpf):
        self.memories.append((state, action, reward, next_state, done, hand_cards, table_cards, trumpf))

    def act(self, input_state):
        if random.uniform(0, 1) >= self.epsilon:
            random_list = [i for i in range(InputHandler.output_size)]
            random.shuffle(random_list)
            return np.random.random_sample(InputHandler.output_size), np.array(random_list)
        first_state = np.expand_dims(input_state[0:self.input_handler_game_network.network_size], axis=0)
        second_state = np.expand_dims(input_state[self.input_handler_game_network.network_size:2 * self.input_handler_game_network.network_size], axis=0)
        third_state = np.expand_dims(input_state[(2 * self.input_handler_game_network.network_size):self.input_handler_game_network.input_size], axis=0)
        network_prediction = self.game_model.predict({'input_1': first_state,
                                                      'input_2': second_state,
                                                      'input_3': third_state})
        act_values = self.choose_output_network(prediction=network_prediction, cards=self.cards)
        return act_values, np.argsort(act_values[0])[::-1]

    def choose_output_network(self, prediction, cards):
        amount_hand_cards = len(cards)
        if 10 > amount_hand_cards > 6:
            return prediction[0]
        elif 0 < amount_hand_cards < 4:
            return prediction[2]
        elif 3 < amount_hand_cards < 7:
            return prediction[1]

    def choose_trumpf(self, geschoben):
        if self.trumpf_model is None:
            allowed = False
            while not allowed:
                trumpf, _ = choose_trumpf(cards=self.cards, geschoben=geschoben)
                allowed = yield trumpf
                if allowed:
                    yield None
        else:
            trumpf_list = self.choose_game_mode(hand_cards=self.cards, geschoben=geschoben)
            for trumpf in trumpf_list:
                yield trumpf

    def choose_game_mode(self, hand_cards, geschoben):
        prediction_input = self.preparation_prediction(hand_cards, geschoben)
        trumpf_calc_list = self.trumpf_model.predict(np.asarray(prediction_input))
        trumpf_choice_list = self.calc_mapping_trumpf(trumpf_calc_list)
        trumpf_choose_list = sorted(trumpf_choice_list.items(), key=operator.itemgetter(1), reverse=True)

        trumpf_list = []
        for tumpf_value in trumpf_choose_list:
            trumpf_list.append(tumpf_value[0])

        # Normaly is the desicion == 'SCHIEBEN' and geschoben == False move the trumpf desicion to the game mate
        # If desicion == 'SCHIEBEN' and geschoben == True remove the first posistion of the list
        if trumpf_list[0].name == 'SCHIEBEN':
            del trumpf_list[0]

        return trumpf_list

    def preparation_prediction(self, hand_cards, geschoben):
        shift = False
        # In the case that geschoben isn't a boolean, so you can change any value to a true otherwise is it false
        # if not geschoben is None:
        #     shift = True
        if isinstance(geschoben, bool) and geschoben:
            shift = True

        handcards = self.fillup_card_list(hand_cards)

        return self.create_trumpf_choose_input(handcards, shift)

    def calc_mapping_trumpf(self, calc_list):
        # The original outputs order of all game mods from training is
        # {'ROSE': '', 'ACORN': '', 'BELL': '', 'SHIELD': '', 'OBE_ABE': '', 'UNDE_UFE': '', 'SCHIEBEN': ''}
        # but you can change the groupings of inputs order if you change also the outputs order of the color.
        # The order of inputs and outputs is interdependent,
        # but the name doesn't matter as long as the groupings remain.
        # choice_list is the output order of all game modes (suits/color and other options)
        order_list = [Trumpf.ROSE, Trumpf.BELL, Trumpf.ACORN, Trumpf.SHIELD, Trumpf.OBE_ABE, Trumpf.UNDE_UFE, Trumpf.SCHIEBEN]
        choice_list = dict(zip(order_list, calc_list[0]))
        return choice_list

    def fillup_card_list(self, cards):
        if not isinstance(cards, list):
            return None
        card_list = []
        if len(cards) > 0:
            for card in cards:
                card_list.append(card)
        return card_list

    def create_trumpf_choose_input(self, handcards, game_type):
        if len(handcards) != 9:
            return None
        inputs = np.zeros((self.input_handler_trumpf_network,))
        for card in handcards:
            if inputs[card_to_index(card)] == 1:
                return None
            inputs[card_to_index(card)] = 1
        if game_type:
            inputs[self.input_handler_trumpf_network - 1] = 1
        else:
            inputs[self.input_handler_trumpf_network - 1] = 0
        return np.reshape(inputs, (1, self.input_handler_trumpf_network))

    def choose_card(self, state=None):
        allowed = False
        self.input_handler_game_network.update_state_choose_card(game_state=state, cards=self.cards, player_id=self.id)
        predictions, prediction_indexes = self.act(self.input_handler_game_network.state)
        card = self.max_of_allowed_cards(state=state, predictions=predictions)
        self.current_memory['action'] = prediction_indexes[0]
        self.current_memory['hand_cards'] = self.cards[:]
        self.current_memory['table_cards'] = [from_string_to_card(entry['card']) for entry in state['table']]
        self.current_memory['trumpf'] = Trumpf[state['trumpf']]
        while not allowed:
            allowed = yield card
            if allowed:
                yield None
            else:
                logger.info('not allowed card!')

    def save_state(self, done):
        if self.previous_memory['used']:
            self.remember(state=self.previous_memory['state'], action=self.previous_memory['action'],
                          reward=self.previous_memory['reward'], done=self.previous_memory['done'],
                          hand_cards=self.previous_memory['hand_cards'],
                          table_cards=self.previous_memory['table_cards'], trumpf=self.previous_memory['trumpf'],
                          next_state=self.current_memory['state'])

        self.previous_memory = self.current_memory.copy()
        if done:
            self.remember(state=self.previous_memory['state'], action=self.previous_memory['action'],
                          reward=self.previous_memory['reward'], done=self.previous_memory['done'],
                          hand_cards=self.previous_memory['hand_cards'],
                          table_cards=self.previous_memory['table_cards'], trumpf=self.previous_memory['trumpf'],
                          next_state=None)
            self.input_handler_game_network.reset()

    def stich_over(self, state=None):
        done = True if len(self.cards) == 0 else False
        last_stich = state['stiche'][-1] if state['stiche'] else None
        self.current_memory['state'] = np.copy(self.input_handler_game_network.state)
        self.current_memory['reward'] = self.calculate_reward(state['teams'], done=done,
                                                              stich=last_stich) if last_stich else 0
        self.current_memory['done'] = done
        self.current_memory['used'] = True
        self.save_state(done=done)
        self.input_handler_game_network.update_state_stich(game_state=state, cards=self.cards, player_id=self.id)

    def calculate_reward(self, teams, done, stich):
        stich_player_id = stich['player_id']
        self.won_stich[stich_player_id] += 1
        team1 = teams[0]['points'] - self.team_points[0]
        team2 = teams[1]['points'] - self.team_points[1]
        self.team_points[0] = teams[0]['points']
        self.team_points[1] = teams[1]['points']
        if done:
            if team1 > team2:
                self.won[0] += 1
                return 2.5
            elif team1 == team2:
                self.won[2] += 1
                return 0
            else:
                self.won[1] += 1
                return -2.5
        if team1 > team2:
            return self.normalize_reward(team1)
        elif team1 == team2:
            return 0
        else:
            return -self.normalize_reward(team2)

    def reset_stats(self):
        self.won = [0, 0, 0]
        self.won_stich = [0, 0, 0, 0]

    def max_of_allowed_cards(self, predictions, state):
        allowed = self.allowed_cards(state=state)
        return max_of_allowed_cards(predictions=predictions, playable_cards=allowed)

    def normalize_reward(self, points):
        return (points - self.min_points) / (self.max_points - self.min_points)

    def denormalize_reward(self, reward):
        return reward * (self.max_points - self.min_points) + self.min_points


def max_of_allowed_cards(predictions, playable_cards):
    indices_of_allowed_cards = [card_to_index(allowed_card) for allowed_card in playable_cards]
    indices_sorted = np.argsort(predictions)[::-1]
    for i in np.nditer(indices_sorted, order='C'):
        try:
            indices_of_allowed_cards.index(i)
            return index_to_card(i)
        except ValueError:
            pass
