from datetime import date, datetime

from pydantic import Field

from app.schemas.common import APIModel


class GCISBatchSource(APIModel):
    dataset_id: str
    title: str
    url: str
    metadata_modified_at: datetime
    local_filename: str


class GCISBatchManifest(APIModel):
    version: str
    as_of: date
    industry_code: str
    industry_mapping_version: str
    provider: str
    coverage: str
    catalog_downloaded_at: date
    sources: list[GCISBatchSource] = Field(min_length=1)


class GCISBatchSourceFileReport(APIModel):
    dataset_id: str
    title: str
    local_filename: str
    byte_size: int = Field(ge=0)
    sha256: str
    reused_existing_file: bool = False


class GCISBatchDownloadReport(APIModel):
    manifest_version: str
    downloaded_at: datetime
    files: list[GCISBatchSourceFileReport]


class GCISBatchImportReport(APIModel):
    manifest_version: str
    as_of: date
    industry_code: str
    industry_mapping_version: str
    provider: str
    coverage: str
    generated_at: datetime
    source_file_count: int = Field(ge=1)
    input_row_count: int = Field(ge=0)
    canonical_company_count: int = Field(ge=0)
    duplicate_row_count: int = Field(ge=0)
    rejected_row_count: int = Field(ge=0)
    rejection_counts: dict[str, int] = Field(default_factory=dict)
    status_description_counts: dict[str, int] = Field(default_factory=dict)
    source_files: list[GCISBatchSourceFileReport]
    canonical_jsonl_sha256: str
