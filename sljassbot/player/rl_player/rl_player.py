import logging
import random
from collections import deque

import numpy as np
from keras.callbacks import ReduceLROnPlateau

from pyschieber.player.base_player import BasePlayer
from pyschieber.trumpf import Trumpf
from pyschieber.card import from_string_to_card
from pyschieber.rules.stich_rules import allowed_cards
from pyschieber.player.greedy_player.trumpf_decision import choose_trumpf
from sljassbot.player.rl_player.input_handler import InputHandler, card_to_index, index_to_card
from sljassbot.player.rl_player.model import build_model

logger = logging.getLogger(__name__)


class RLPlayer(BasePlayer):
    max_points = 55
    min_points = -55

    def __init__(self, name, model_path, rounds=1):
        super().__init__(name=name)
        self.input_handler = InputHandler()
        self.n_samples = 20 * rounds
        self.memories = deque([], maxlen=2 * self.n_samples)
        self.model = build_model(model_path=model_path)
        self.model_t = build_model(model_path=model_path)
        self.loss = []
        self.epsilon = 0.95  # exploration rate
        self.penalty = 0.
        self.gamma = 0.95
        self.callbacks = [ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.01)]
        self.won = 0
        self.lost = 0
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
        state = np.expand_dims(input_state, axis=0)
        act_values = self.model.predict(state)
        return act_values, np.argsort(act_values[0])[::-1]

    def replay(self):
        for state, action, reward, next_state, done, hand_cards, table_cards, trumpf in self.get_batch():
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)
            target = reward
            if not done:
                # predictions = self.model_t.predict(next_state)
                # value = self.replay_max_allowed_value(predictions, hand_cards, table_cards, trumpf)
                target = reward + self.gamma * np.amax(self.model_t.predict(next_state)[0])
                # target = reward + self.gamma * value
            target_f = self.model.predict(state)
            target_f[0][action] = target
            history = self.model.fit(state, target_f, epochs=1, verbose=0, callbacks=self.callbacks)
            self.loss += history.history['loss']
        self.update_target()

    def replay_max_allowed_value(self, predictions, hand_cards, table_cards, trumpf):
        playable_cards = allowed_cards(hand_cards=hand_cards, table_cards=table_cards, trumpf=trumpf)
        card = max_of_allowed_cards(predictions=predictions, playable_cards=playable_cards)
        index = card_to_index(card=card)
        return predictions[0][index]

    def update_target(self):
        weights = self.model.get_weights()
        self.model_t.set_weights(weights)

    def get_batch(self):
        n_samples = min(self.n_samples, len(self.memories))
        samples = random.sample(self.memories, n_samples)
        return samples

    def choose_trumpf(self, geschoben):
        allowed = False
        while not allowed:
            trumpf, _ = choose_trumpf(cards=self.cards, geschoben=geschoben)
            allowed = yield trumpf
            if allowed:
                yield None

    def choose_card(self, state=None):
        allowed = False
        self.input_handler.update_state_choose_card(game_state=state, cards=self.cards, player_id=self.id)
        predictions, prediction_indexes = self.act(self.input_handler.state)
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
            # print_state(self.previous_memory['state'], 0)
            self.remember(state=self.previous_memory['state'], action=self.previous_memory['action'],
                          reward=self.previous_memory['reward'], done=self.previous_memory['done'],
                          hand_cards=self.previous_memory['hand_cards'],
                          table_cards=self.previous_memory['table_cards'], trumpf=self.previous_memory['trumpf'],
                          next_state=None)
            self.input_handler.reset()
            # print('Number of played cards: ',
            #      count_played_cards(input_state=self.previous_memory['state'], player_id=self.id))

    def stich_over(self, state=None):
        done = True if len(self.cards) == 0 else False
        last_stich = state['stiche'][-1] if state['stiche'] else None
        self.current_memory['state'] = np.copy(self.input_handler.state)
        self.current_memory['reward'] = self.calculate_reward(state['teams'], done=done,
                                                              stich=last_stich) if last_stich else 0
        self.current_memory['done'] = done
        self.current_memory['used'] = True
        self.save_state(done=done)
        self.input_handler.update_state_stich(game_state=state, cards=self.cards, player_id=self.id)

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
