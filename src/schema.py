from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class Severity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class FoodAction(str, Enum):
    AVOID = "avoid"
    CAUTION = "caution"
    OK = "ok"


class DosageInstruction(BaseModel):
    time_of_day: str = Field(description="morning / afternoon / evening / bedtime")
    amount: str = Field(description="e.g. 3mg, 1 tablet")
    with_food: bool = Field(description="True if must be taken with food")
    notes: Optional[str] = Field(default=None)


class SideEffect(BaseModel):
    name: str
    severity: Severity
    description: str


class FoodInteraction(BaseModel):
    substance: str
    action: FoodAction
    reason: str


class Warning(BaseModel):
    text: str
    applies_to: List[str]


class DrugInfo(BaseModel):
    drug_name: str
    active_ingredient: str
    drug_class: str
    dosage_instructions: List[DosageInstruction]
    side_effects: List[SideEffect]
    food_interactions: List[FoodInteraction] = Field(default_factory=list)
    warnings: List[Warning] = Field(default_factory=list)
    contraindications: List[str] = Field(default_factory=list)
    emergency_signs: List[str] = Field(default_factory=list)


class UserProfile(BaseModel):
    age_group: str = Field(description="child / adult / elderly")
    sex: str = Field(default="prefer_not_to_say", description="male / female / prefer_not_to_say")
    pregnant: bool = False
    breastfeeding: bool = False
    liver_issue: bool = False
    kidney_issue: bool = False
    heart_condition: bool = False
    diabetes: bool = False
    hypertension: bool = False
    asthma: bool = False
    other_conditions: str = Field(default="", description="free text for other conditions")
    other_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
