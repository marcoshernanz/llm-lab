from __future__ import annotations

from pathlib import Path
import tomllib
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError
from pydantic import ValidationInfo
from pydantic import field_validator
from pydantic import model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunConfig(ConfigModel):
    seed: int = Field(gt=0)


class DataConfig(ConfigModel):
    dataset_path: Path
    tokenizer_path: Path
    context_tokens: int = Field(gt=0)
    text_limit: int | None = Field(default=None, gt=0)

    @field_validator("dataset_path", mode="before")
    @classmethod
    def resolve_dataset_path(cls, value: object, info: ValidationInfo) -> Path:
        return _resolve_path("data.dataset_path", value, info)

    @field_validator("tokenizer_path", mode="before")
    @classmethod
    def resolve_tokenizer_path(cls, value: object, info: ValidationInfo) -> Path:
        return _resolve_path("data.tokenizer_path", value, info)

    @model_validator(mode="after")
    def validate_paths_and_ratio(self) -> DataConfig:
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")
        if not self.tokenizer_path.exists():
            raise ValueError(f"Tokenizer path does not exist: {self.tokenizer_path}")
        return self


class ModelConfig(ConfigModel):
    embedding_dim: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    num_decoder_blocks: int = Field(gt=0)
    hidden_dim: int = Field(gt=0)


class OptimizerConfig(ConfigModel):
    learning_rate: float = Field(gt=0)


class TrainConfig(ConfigModel):
    steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    eval_batch_size: int = Field(gt=0)
    sample_tokens: int = Field(gt=0)


class TrainingConfig(ConfigModel):
    run: RunConfig
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    train: TrainConfig

    @model_validator(mode="after")
    def validate_cross_section_constraints(self) -> TrainingConfig:
        if self.model.embedding_dim % self.model.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        return self


def load_config(path: Path) -> TrainingConfig:
    config_path = path.resolve()
    payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    try:
        return TrainingConfig.model_validate(payload, context={"base_dir": config_path.parent})
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def _resolve_path(name: str, value: object, info: ValidationInfo) -> Path:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{name} must be a non-empty string.")
    path = Path(value).expanduser()
    base_dir = info.context.get("base_dir") if isinstance(info.context, dict) else None
    if not isinstance(base_dir, Path):
        raise ValueError("Config validation context is missing base_dir.")
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path
