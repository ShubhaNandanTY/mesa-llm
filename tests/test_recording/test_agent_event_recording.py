"""Tests that LLMAgent observations and messages reach the SimulationRecorder."""

import io
import logging
from unittest.mock import Mock

import pytest
from mesa.model import Model
from mesa.space import MultiGrid
from rich.console import Console

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_memory import ShortTermMemory
from mesa_llm.recording.agent_analysis import AgentViewer
from mesa_llm.recording.record_model import record_model


@pytest.fixture
def recorded_model(temp_dir, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    # temp_dir is gone by interpreter exit, so skip @record_model's atexit auto-save
    monkeypatch.setattr(
        "mesa_llm.recording.record_model.atexit.register", lambda func: None
    )

    @record_model(output_dir=str(temp_dir))
    class DummyModel(Model):
        def __init__(self):
            super().__init__(rng=42)
            self.grid = MultiGrid(3, 3, torus=False)
            self.sender, self.recipient = LLMAgent.create_agents(
                self,
                n=2,
                reasoning=lambda agent: None,
                vision=-1,
                internal_state=["calm"],
            ).to_list()
            for agent, pos in ((self.sender, (0, 0)), (self.recipient, (1, 1))):
                self.grid.place_agent(agent, pos)
                agent.memory = ShortTermMemory(agent=agent, n=5, display=False)

    return DummyModel()


def agent_events(model, event_type):
    return [e for e in model.recorder.events if e.event_type == event_type]


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_observation_is_recorded(recorded_model, use_async):
    sender = recorded_model.sender

    if use_async:
        obs = await sender.agenerate_obs()
    else:
        obs = sender.generate_obs()

    (event,) = agent_events(recorded_model, "observation")
    assert event.agent_id == sender.unique_id
    assert event.content == {
        "self_state": obs.self_state,
        "local_state": obs.local_state,
    }
    assert event.metadata == {"source": "LLMAgent.generate_obs"}


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_message_is_recorded_once_on_the_sender(recorded_model, use_async):
    sender, recipient = recorded_model.sender, recorded_model.recipient

    if use_async:
        await sender.asend_message("hello", [recipient])
    else:
        sender.send_message("hello", [recipient])

    (event,) = agent_events(recorded_model, "message")
    assert event.agent_id == sender.unique_id
    assert event.content == {"message": "hello", "recipient_ids": [recipient.unique_id]}
    assert event.metadata == {"source": "LLMAgent.send_message"}


def test_message_records_only_delivered_recipients(recorded_model):
    sender, recipient = recorded_model.sender, recorded_model.recipient
    no_memory = Mock(spec=["unique_id"], unique_id=99)

    sender.send_message("hello", [recipient, no_memory, sender])

    (event,) = agent_events(recorded_model, "message")
    assert event.content["recipient_ids"] == [recipient.unique_id]


def test_recorded_observation_is_a_snapshot(recorded_model):
    sender = recorded_model.sender

    sender.generate_obs()
    sender.internal_state.append("angry")

    (event,) = agent_events(recorded_model, "observation")
    assert event.content["self_state"]["internal_state"] == ["calm"]


def test_recorder_failure_does_not_interrupt_the_step(recorded_model, caplog):
    sender, recipient = recorded_model.sender, recorded_model.recipient
    sender.recorder = Mock()
    sender.recorder.record_event.side_effect = RuntimeError("disk full")

    with caplog.at_level(logging.ERROR, logger="mesa_llm.llm_agent"):
        obs = sender.generate_obs()
        status = sender.send_message("hello", [recipient])

    assert obs.self_state["internal_state"] == ["calm"]
    assert status == f"sent message 'hello' to [{recipient.unique_id}]"
    assert "message" in recipient.memory.step_content
    assert "Recording observation event" in caplog.text
    assert "Recording message event" in caplog.text


def test_no_recorder_is_a_no_op(recorded_model):
    sender, recipient = recorded_model.sender, recorded_model.recipient
    sender.recorder = None

    sender.generate_obs()
    sender.send_message("hello", [recipient])

    assert agent_events(recorded_model, "observation") == []
    assert agent_events(recorded_model, "message") == []


def test_agent_viewer_shows_recorded_conversation(recorded_model):
    sender, recipient = recorded_model.sender, recorded_model.recipient
    sender.generate_obs()
    recipient.generate_obs()
    sender.send_message("hello there", [recipient])
    path = recorded_model.save_recording("conversation.json")

    viewer = AgentViewer(str(path))
    output = io.StringIO()
    viewer.console = Console(file=output, width=200)
    viewer.view_agent_conversations(recipient.unique_id)
    viewer.view_agent_summary(sender.unique_id)

    text = output.getvalue()
    assert f"From agent {sender.unique_id}: hello there" in text
    assert "Messages Sent" in text
    assert sender.unique_id in viewer.agent_events
    assert {e["event_type"] for e in viewer.agent_events[sender.unique_id]} == {
        "observation",
        "message",
    }
