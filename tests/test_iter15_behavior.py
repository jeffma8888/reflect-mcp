"""Black-box behavioral tests for iter-15: GateConfig dataclass→Pydantic migration.

Covers all 7 Expected Behaviors from pm.md:
  1. GateConfig() defaults: field_threshold=0.70, overall_threshold=0.75
  2. GateConfig(field_threshold=1.5) raises pydantic.ValidationError (le=1.0)
  3. GateConfig(overall_threshold=-0.1) raises pydantic.ValidationError (ge=0.0)
  4. Frozen semantics: assigning config.field_threshold = 0.5 raises error
  5. Existing gate decision tests pass unchanged (verified by full suite run)
  6. GateConfig is a Pydantic BaseModel, not a dataclass
  7. Full quality-check suite passes (verified by full suite run)

Isolation contract: tests drive the public GateConfig interface and assert
observable output. No implementation source was read.
"""

from __future__ import annotations

import dataclasses

import pytest
from pydantic import BaseModel, ValidationError

from reflect_mcp.gate import GateConfig


# ---------------------------------------------------------------------------
# EB-1: Default values
# ---------------------------------------------------------------------------

class TestDefaultValues:
    def test_default_field_threshold(self):
        config = GateConfig()
        assert config.field_threshold == pytest.approx(0.70)

    def test_default_overall_threshold(self):
        config = GateConfig()
        assert config.overall_threshold == pytest.approx(0.75)

    def test_defaults_both_present(self):
        config = GateConfig()
        assert config.field_threshold == pytest.approx(0.70)
        assert config.overall_threshold == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# EB-2: field_threshold > 1.0 raises ValidationError
# ---------------------------------------------------------------------------

class TestFieldThresholdUpperBound:
    def test_field_threshold_above_one_raises(self):
        with pytest.raises(ValidationError):
            GateConfig(field_threshold=1.5)

    def test_field_threshold_exactly_one_ok(self):
        config = GateConfig(field_threshold=1.0)
        assert config.field_threshold == pytest.approx(1.0)

    def test_field_threshold_slightly_above_one_raises(self):
        with pytest.raises(ValidationError):
            GateConfig(field_threshold=1.0001)


# ---------------------------------------------------------------------------
# EB-3: overall_threshold < 0.0 raises ValidationError
# ---------------------------------------------------------------------------

class TestOverallThresholdLowerBound:
    def test_overall_threshold_negative_raises(self):
        with pytest.raises(ValidationError):
            GateConfig(overall_threshold=-0.1)

    def test_overall_threshold_zero_ok(self):
        config = GateConfig(overall_threshold=0.0)
        assert config.overall_threshold == pytest.approx(0.0)

    def test_overall_threshold_slightly_negative_raises(self):
        with pytest.raises(ValidationError):
            GateConfig(overall_threshold=-0.001)


# ---------------------------------------------------------------------------
# EB-4: Frozen (immutable) semantics
# ---------------------------------------------------------------------------

class TestFrozenSemantics:
    def test_field_threshold_assignment_raises(self):
        config = GateConfig()
        with pytest.raises(ValidationError):
            config.field_threshold = 0.5

    def test_overall_threshold_assignment_raises(self):
        config = GateConfig()
        with pytest.raises(ValidationError):
            config.overall_threshold = 0.5


# ---------------------------------------------------------------------------
# EB-6: GateConfig is a Pydantic BaseModel, not a dataclass
# ---------------------------------------------------------------------------

class TestPydanticNotDataclass:
    def test_is_pydantic_basemodel(self):
        assert issubclass(GateConfig, BaseModel)

    def test_is_not_a_dataclass(self):
        assert not dataclasses.is_dataclass(GateConfig)

    def test_instance_is_pydantic(self):
        config = GateConfig()
        assert isinstance(config, BaseModel)


# ---------------------------------------------------------------------------
# EB-2/EB-3 combined: both bounds for both fields
# ---------------------------------------------------------------------------

class TestCrossBoundaryValidation:
    def test_field_threshold_below_zero_raises(self):
        with pytest.raises(ValidationError):
            GateConfig(field_threshold=-0.1)

    def test_overall_threshold_above_one_raises(self):
        with pytest.raises(ValidationError):
            GateConfig(overall_threshold=1.5)

    def test_both_valid_boundary_values(self):
        config = GateConfig(field_threshold=0.0, overall_threshold=1.0)
        assert config.field_threshold == pytest.approx(0.0)
        assert config.overall_threshold == pytest.approx(1.0)

    def test_both_invalid_raises(self):
        with pytest.raises(ValidationError):
            GateConfig(field_threshold=2.0, overall_threshold=-1.0)
