import numpy as np
import pytest
import torch

from src.models.dqn_agent import DQN, DQNAgent, Experience


@pytest.fixture
def dqn_model():
    state_size = 10
    action_size = 3
    return DQN(state_size, action_size)


@pytest.fixture
def dqn_agent():
    state_size = 10
    action_size = 3
    return DQNAgent(state_size, action_size)


# DQN Network Tests
def test_network_architecture(dqn_model):
    """Test if the DQN architecture is correctly initialized"""
    assert dqn_model.fc1.in_features == 10
    assert dqn_model.fc1.out_features == 64
    assert dqn_model.fc2.in_features == 64
    assert dqn_model.fc2.out_features == 64
    assert dqn_model.fc3.in_features == 64
    assert dqn_model.fc3.out_features == 3


def test_forward_pass(dqn_model):
    """Test if forward pass produces expected output shape"""
    batch_size = 32
    x = torch.randn(batch_size, 10)
    output = dqn_model(x)
    assert output.shape == (batch_size, 3)


# DQN Agent Tests
def test_agent_initialization(dqn_agent):
    """Test if agent is correctly initialized"""
    assert dqn_agent.state_size == 10
    assert dqn_agent.action_size == 3
    assert dqn_agent.epsilon == 1.0
    assert len(dqn_agent.memory) == 0


def test_remember(dqn_agent):
    """Test if experiences are correctly stored in memory"""
    state = np.random.random(10)
    action = 1
    reward = 0.5
    next_state = np.random.random(10)
    done = False

    initial_memory_size = len(dqn_agent.memory)
    dqn_agent.remember(
        Experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
    )
    assert len(dqn_agent.memory) == initial_memory_size + 1


def test_act_explore(dqn_agent):
    """Test exploration in act method"""
    dqn_agent.epsilon = 1.0  # Force exploration
    state = np.random.random(10)
    action = dqn_agent.act(state)
    assert 0 <= action < dqn_agent.action_size


def test_act_exploit(dqn_agent):
    """Test exploitation in act method"""
    dqn_agent.epsilon = 0.0  # Force exploitation
    state = np.random.random(10)
    action = dqn_agent.act(state)
    assert 0 <= action < dqn_agent.action_size


def test_epsilon_decay(dqn_agent):
    """Test if epsilon decays correctly during replay"""
    initial_epsilon = dqn_agent.epsilon

    # Add experiences to memory
    for _ in range(32):
        state = np.random.random(10)
        action = 1
        reward = 0.5
        next_state = np.random.random(10)
        done = False
        dqn_agent.remember(
            Experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        )

    dqn_agent.replay(32)
    assert dqn_agent.epsilon < initial_epsilon


def test_target_model_update(dqn_agent):
    """Test if target model is correctly updated"""
    # Get initial parameters
    initial_params = [param.clone().detach() for param in dqn_agent.target_model.parameters()]

    # Make some updates to the main model
    state = np.random.random(10)
    action = 1
    reward = 0.5
    next_state = np.random.random(10)
    done = False
    dqn_agent.remember(
        Experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
    )
    dqn_agent.replay(1)

    # Update target model
    dqn_agent.update_target_model()

    # Check if parameters are updated
    for target_param, initial_param in zip(dqn_agent.target_model.parameters(), initial_params):
        assert not torch.equal(target_param, initial_param)
