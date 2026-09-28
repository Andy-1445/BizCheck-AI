import json
from pathlib import Path

from app.schemas.llm_analysis import CompanyAnalysisLLMInput
from app.services.llm_prompt import build_company_analysis_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_EXAMPLE = (
    PROJECT_ROOT / "samples" / "llm" / "company-analysis-input-v1.example.json"
)
OUTPUT_EXAMPLE = (
    PROJECT_ROOT / "samples" / "llm" / "company-analysis-prompt-v1.example.json"
)


def main() -> None:
    analysis_input = CompanyAnalysisLLMInput.model_validate_json(
        INPUT_EXAMPLE.read_text(encoding="utf-8")
    )
    prompt_package = build_company_analysis_prompt(analysis_input)
    OUTPUT_EXAMPLE.write_text(
        json.dumps(
            prompt_package.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
