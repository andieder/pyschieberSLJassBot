import pytest
from pyschieber.card import Card
from pyschieber.suit import Suit

from sljassbot.player.rl_player.input_handler import card_to_index, index_to_card


@pytest.mark.parametrize("card, index", [
    (Card(suit=Suit.ROSE, value=6), 0),
    (Card(suit=Suit.ROSE, value=7), 1),
    (Card(suit=Suit.BELL, value=6), 9),
    (Card(suit=Suit.BELL, value=7), 10),
])
def test_card_to_index(card, index):
    assert card_to_index(card=card) == index


@pytest.mark.parametrize("card, index", [
    (Card(suit=Suit.ROSE, value=6), 0),
    (Card(suit=Suit.ROSE, value=7), 1),
    (Card(suit=Suit.BELL, value=6), 9),
    (Card(suit=Suit.BELL, value=7), 10),
])
def test_index_to_card(card, index):
    assert index_to_card(index=index) == card
